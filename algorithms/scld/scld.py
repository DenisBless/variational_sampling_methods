from functools import partial

import distrax
import jax
import jax.numpy as jnp
import chex
import optax
from flax.training import train_state

import algorithms.common.types as tp
from algorithms.common import markov_kernel
from algorithms.common.models.pisgrad_net import PISGRADNet
from algorithms.scld.is_weights import sub_traj_is_weights, update_samples_log_weights, get_lnz_elbo_increment
from algorithms.scld.loss_fns import get_loss_fn
from algorithms.scld.prioritised_buffer import build_prioritised_buffer
from algorithms.scld.scld_eval import eval_scld
from algorithms.scld.scld_utils import GeometricAnnealingSchedule, print_results, gradient_step, flattened_traversal
import wandb

Array = tp.Array
FlowApply = tp.FlowApply
FlowParams = tp.FlowParams
LogDensityByStep = tp.LogDensityByStep
LogDensityNoStep = tp.LogDensityNoStep
MarkovKernelApply = tp.MarkovKernelApply
AcceptanceTuple = tp.AcceptanceTuple
RandomKey = tp.RandomKey
Samples = tp.Samples
assert_equal_shape = chex.assert_equal_shape
assert_trees_all_equal_shapes = chex.assert_trees_all_equal_shapes


def inner_step_simulate(key,
                        model_state,
                        params,
                        samples,
                        log_weights,
                        sim_tuple,
                        markov_kernel_apply,
                        sub_traj,
                        config,
                        allow_smc):
    key, key_gen = jax.random.split(key)
    keys = jax.random.split(key, samples.shape[0])
    log_is_weights, aux = sub_traj_is_weights(keys, samples, model_state, params,
                                              sim_tuple, sub_traj)
    model_samples, target_log_probs = aux

    increments = get_lnz_elbo_increment(log_is_weights, log_weights)

    sub_traj_start_point, sub_traj_end_point, sub_traj_idx, sub_traj_length = sub_traj

    key, key_gen = jax.random.split(key_gen)
    next_samples, next_log_weights, acceptance_tuple = update_samples_log_weights(
        samples=model_samples, log_is_weights=log_is_weights, markov_kernel_apply=markov_kernel_apply,
        log_weights=log_weights, step=sub_traj_end_point[0], key=key,
        use_reweighting=allow_smc, use_resampling=config.use_resampling * allow_smc, resampler=config.resampler,
        use_markov=config.use_markov * allow_smc, resample_threshold=config.resample_threshold)

    if config.use_markov:
        (log_density_per_step, noise_schedule, total_steps) = sim_tuple
        target_log_probs = log_density_per_step(sub_traj_end_point[0], next_samples)[:, None]

    return next_samples, next_log_weights, increments, target_log_probs


def simulate(key_gen,
             model_state,
             params,
             initial_sampler,
             log_density_per_step,
             markov_kernel_apply,
             traj,
             config,
             allow_smc=True):
    key, key_gen = jax.random.split(key_gen)
    initial_samples = initial_sampler(seed=key, sample_shape=(config.batch_size,))
    initial_log_weights = -jnp.log(config.batch_size) * jnp.ones(config.batch_size)

    (n_sub_traj, sub_traj_start_points, sub_traj_end_points, sub_traj_indices, sub_traj_length) = traj

    key, key_gen = jax.random.split(key_gen)
    sub_traj_keys = jax.random.split(key, n_sub_traj)
    sim_tuple = (log_density_per_step, config.noise_schedule, config.num_steps)

    # Define initial state and per step inputs for scan step
    initial_state = (initial_samples, initial_log_weights)
    per_step_inputs = (sub_traj_keys, sub_traj_start_points, sub_traj_end_points, sub_traj_indices)

    def scan_step(state, per_step_input):
        samples, log_weights = state
        key, sub_traj_start_point, sub_traj_end_point, sub_traj_idx = per_step_input
        sub_traj = (sub_traj_start_point, sub_traj_end_point, sub_traj_idx, sub_traj_length)
        next_samples, next_log_weights, increments, target_log_probs = inner_step_simulate(key, model_state, params,
                                                                                           samples, log_weights,
                                                                                           sim_tuple,
                                                                                           markov_kernel_apply,
                                                                                           sub_traj, config,
                                                                                           allow_smc)

        next_state = (next_samples, next_log_weights)
        per_step_output = (next_samples, increments, target_log_probs)
        return next_state, per_step_output

    final_state, per_step_outputs = jax.lax.scan(scan_step, initial_state, per_step_inputs)
    samples, (lnz_incs, elbo_incs), sub_traj_target_log_probs = per_step_outputs
    lnz, elbo = jnp.sum(lnz_incs), jnp.sum(elbo_incs)
    return jnp.concatenate([jnp.expand_dims(initial_samples, 0), samples], axis=0), \
        jnp.concatenate([jnp.ones((1, config.batch_size)), sub_traj_target_log_probs[:, :, 0]], axis=0), \
        (lnz, elbo)


def scld_trainer(cfg, target):
    # Initialization
    key_gen = jax.random.PRNGKey(cfg.seed)
    dim = target.dim
    alg_cfg = cfg.algorithm

    target_samples = target.sample(seed=jax.random.PRNGKey(cfg.seed), sample_shape=(cfg.eval_samples,))
    buffer = build_prioritised_buffer(dim, alg_cfg.n_sub_traj,
                                      jnp.array(alg_cfg.buffer.max_length_in_batches * alg_cfg.batch_size, dtype=int),
                                      jnp.array(alg_cfg.buffer.min_length_in_batches * alg_cfg.batch_size, dtype=int),
                                      sample_with_replacement=alg_cfg.buffer.sample_with_replacement,
                                      prioritized=alg_cfg.buffer.prioritized)

    # Compute boundaries of sub-trajectories
    sub_traj_length = alg_cfg.num_steps // alg_cfg.n_sub_traj
    num_transitions = alg_cfg.num_steps + 1  # todo check if this is correct
    n_sub_traj = alg_cfg.n_sub_traj
    sub_traj_start_points = jnp.array([[t * sub_traj_length] for t in range(n_sub_traj)])
    sub_traj_end_points = jnp.array([[(t + 1) * sub_traj_length] for t in range(n_sub_traj)])
    sub_traj_indices = jnp.arange(n_sub_traj)
    traj = (n_sub_traj, sub_traj_start_points, sub_traj_end_points, sub_traj_indices, sub_traj_length)

    # Define the model
    model = PISGRADNet(**alg_cfg.model)
    key, key_gen = jax.random.split(key_gen)
    params = model.init(key, jnp.ones([alg_cfg.batch_size, dim]),
                        jnp.ones([alg_cfg.batch_size, 1]),
                        jnp.ones([alg_cfg.batch_size, dim]))

    if alg_cfg.loss in ['rev_tb', 'fwd_tb']:
        additional_params = {'logZ': jnp.ones(n_sub_traj) * alg_cfg.init_logZ}
        params['params'] = {**params['params'], **additional_params}

    optimizer = optax.chain(
        optax.clip(alg_cfg.grad_clip) if alg_cfg.grad_clip > 0 else optax.identity(),
        optax.masked(optax.adam(learning_rate=alg_cfg.step_size),
                     mask=flattened_traversal(lambda path, _: path[-1] != 'logZ')),
        optax.masked(optax.sgd(learning_rate=alg_cfg.logZ_step_size),
                     mask=flattened_traversal(lambda path, _: path[-1] == 'logZ')) if alg_cfg.loss in ['rev_tb',
                                                                                                       'fwd_tb'] else optax.identity(),
    )

    model_state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

    initial_density = distrax.MultivariateNormalDiag(jnp.zeros(dim),
                                                     jnp.ones(dim) * alg_cfg.init_std)

    log_density_per_step = GeometricAnnealingSchedule(initial_density.log_prob, target.log_prob,
                                                      num_transitions, alg_cfg.target_grad_clip)
    markov_kernel_by_step = markov_kernel.MarkovTransitionKernel(alg_cfg.mcmc, log_density_per_step,
                                                                 num_transitions)

    simulate_short = jax.jit(partial(simulate,
                                     initial_sampler=initial_density.sample,
                                     log_density_per_step=log_density_per_step,
                                     markov_kernel_apply=markov_kernel_by_step,
                                     traj=traj, config=alg_cfg, allow_smc=True))

    simulate_short_no_smc = jax.jit(partial(simulate,
                                            initial_sampler=initial_density.sample,
                                            log_density_per_step=log_density_per_step,
                                            markov_kernel_apply=markov_kernel_by_step, traj=traj,
                                            config=alg_cfg, allow_smc=False))

    sim_tuple = (log_density_per_step, alg_cfg.noise_schedule, num_transitions)
    sub_traj_loss = get_loss_fn(alg_cfg.loss)

    def sub_traj_loss_short(keys, samples, next_samples, model_state, params, sub_traj_start_points,
                            sub_traj_end_points, sub_traj_indices):
        return sub_traj_loss(keys, samples, next_samples, model_state, params, sim_tuple,
                             sub_traj_start_points, sub_traj_end_points, sub_traj_indices, sub_traj_length)

    loss_fn = jax.vmap(jax.value_and_grad(jax.jit(sub_traj_loss_short), 4, has_aux=True),
                       in_axes=(0, 0, 0, None, None, 0, 0, 0))

    key, key_gen = jax.random.split(key_gen)
    eval_fn = eval_scld(simulate_short, simulate_short_no_smc, target, target_samples, cfg)

    key, key_gen = jax.random.split(key_gen)

    # if cfg.use_wandb:
    #     wandb.log(eval_fn(model_state, model_state.params, key))
    # return
    init_samples, sub_traj_target_log_probs, _ = simulate_short(key, model_state, params)
    buffer_state = buffer.init(init_samples, sub_traj_target_log_probs)

    logger = {}
    eval_freq = alg_cfg.n_sim * alg_cfg.n_updates_per_sim // cfg.n_evals

    for i in range(alg_cfg.n_sim):
        key, key_gen = jax.random.split(key_gen)
        sim_samples, sub_traj_target_log_probs, (lnz_est, elbo_est) = simulate_short(key, model_state,
                                                                                     model_state.params)
        # print(f'lnz {lnz_est}, elbo {elbo_est}')
        buffer_state = buffer.add(sim_samples, sub_traj_target_log_probs, buffer_state=buffer_state)
        for j in range(alg_cfg.n_updates_per_sim):
            key, key_gen = jax.random.split(key_gen)
            buffer_samples = buffer.sample(key=key, buffer_state=buffer_state, batch_size=alg_cfg.batch_size)

            key, key_gen = jax.random.split(key_gen)
            keys = jax.random.split(key, (n_sub_traj, alg_cfg.batch_size,))

            (per_sample_loss, aux), grads_all = loss_fn(keys, buffer_samples[:-1], buffer_samples[1:], model_state,
                                                        model_state.params,
                                                        sub_traj_start_points, sub_traj_end_points, sub_traj_indices)

            model_state = gradient_step(model_state, grads_all)

            if cfg.use_wandb:
                wandb.log({'loss_hist': per_sample_loss})
                wandb.log({'loss': jnp.mean(per_sample_loss)})

        if i % eval_freq == 0:
            # target.visualise(buffer_samples[10], show=True)
            key, key_gen = jax.random.split(key_gen)
            logger.update(eval_fn(model_state, model_state.params, key))
            logger["stats/step"] = i

            print_results(i, logger, cfg)
            print(f'Loss: {jnp.mean(per_sample_loss)}')

            if cfg.use_wandb:
                wandb.log(logger)
