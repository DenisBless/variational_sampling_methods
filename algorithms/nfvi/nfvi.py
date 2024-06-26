import functools
from time import time

import wandb
import algorithms.common.types as tp
import chex
import jax
import jax.numpy as jnp
import optax

from algorithms.common.eval_methods.tractable_density_methods import get_eval_fn
from targets.base_target import Target
from utils.print_util import print_results

Array = jnp.ndarray
UpdateFn = tp.UpdateFn
OptState = tp.OptState
FlowParams = tp.FlowParams
FlowApply = tp.FlowApply
LogDensityNoStep = tp.LogDensityNoStep
InitialSampler = tp.InitialSampler
RandomKey = tp.RandomKey
assert_equal_shape = chex.assert_equal_shape
AlgoResultsTuple = tp.AlgoResultsTuple
ParticleState = tp.ParticleState

assert_trees_all_equal_shapes = chex.assert_trees_all_equal_shapes


def vi_free_energy(flow_params: FlowParams,
                   key: RandomKey,
                   initial_sampler: InitialSampler,
                   initial_density: LogDensityNoStep,
                   final_density: LogDensityNoStep,
                   flow_apply: FlowApply,
                   cfg):
    """The variational free energy used in VI with normalizing flows."""
    samples = initial_sampler(seed=key, sample_shape=(cfg.algorithm.batch_size,))
    transformed_samples, log_det_jacs = flow_apply(flow_params, samples)
    assert_trees_all_equal_shapes(transformed_samples, samples)
    log_density_target = final_density(transformed_samples)
    log_density_initial = initial_density(samples)
    assert_equal_shape([log_density_initial, log_density_target])
    log_density_approx = log_density_initial - log_det_jacs
    assert_equal_shape([log_density_approx, log_density_initial])
    free_energies = log_density_approx - log_density_target
    free_energy = jnp.mean(free_energies)
    return free_energy


def outer_loop_vi(initial_sampler: InitialSampler,
                  opt_update: UpdateFn,
                  opt_init_state: OptState,
                  flow_init_params: FlowParams,
                  flow_apply: FlowApply,
                  flow_inverse_apply: FlowApply,
                  initial_log_density: LogDensityNoStep,
                  target: Target,
                  cfg,
                  ):

    def eval_nfvi(key: RandomKey):
        """Estimate log normalizing constant using naive importance sampling."""
        samples = initial_sampler(seed=key, sample_shape=(cfg.eval_samples,))
        transformed_samples, log_det_jacs = flow_apply(flow_params, samples)
        assert_trees_all_equal_shapes(transformed_samples, samples)
        log_density_target = target_log_density(transformed_samples)
        log_density_initial = initial_log_density(samples)
        assert_equal_shape([log_density_initial, log_density_target])
        log_density_approx = log_density_initial - log_det_jacs
        log_ratio = log_density_target - log_density_approx

        if cfg.compute_forward_metrics and (target_samples is not None):
            fwd_target_log_p = target_log_density(target_samples)
            prior_samples, inv_log_det_jacs = flow_inverse_apply(flow_params, target_samples)
            fwd_model_log_p = initial_log_density(prior_samples) + inv_log_det_jacs
            fwd_log_ratio = fwd_target_log_p - fwd_model_log_p
            logger = eval_fn(samples, log_ratio, log_density_target, fwd_log_ratio)
        else:
            logger = eval_fn(samples, log_ratio, log_density_target, None)

        return logger

    alg_cfg = cfg.algorithm
    eval_freq = max(alg_cfg.iters // cfg.n_evals, 1)

    vi_free_energy_short = functools.partial(vi_free_energy,
                                             initial_sampler=initial_sampler,
                                             initial_density=initial_log_density,
                                             final_density=target.log_prob,
                                             flow_apply=flow_apply,
                                             cfg=cfg)

    free_energy_and_grad = jax.jit(jax.value_and_grad(vi_free_energy_short))

    flow_params = flow_init_params
    opt_state = opt_init_state

    @jax.jit
    def nfvi_update(curr_key, curr_flow_params, curr_opt_state):
        subkey, curr_key = jax.random.split(curr_key)
        new_free_energy, flow_grads = free_energy_and_grad(curr_flow_params,
                                                           subkey)
        updates, new_opt_state = opt_update(flow_grads,
                                            curr_opt_state)
        new_flow_params = optax.apply_updates(curr_flow_params,
                                              updates)
        return curr_key, new_flow_params, new_free_energy, new_opt_state

    key = jax.random.PRNGKey(cfg.seed)

    target_log_density = target.log_prob
    target_samples = target.sample(jax.random.PRNGKey(0), (cfg.eval_samples,))

    eval_fn, logger = get_eval_fn(cfg, target, target_samples)

    timer = 0

    for step in range(alg_cfg.iters):
        with jax.profiler.StepTraceAnnotation('train', step_num=step):
            iter_time = time()
            key, flow_params, curr_free_energy, opt_state = nfvi_update(key, flow_params, opt_state)
            timer += time() - iter_time

            if step % eval_freq == 0:
                key, subkey = jax.random.split(key)
                logger = eval_nfvi(subkey)
                logger["stats/step"].append(step)
                logger["stats/wallclock"].append(timer)
                logger["stats/nfe"].append((step + 1) * alg_cfg.batch_size)

                print_results(step, logger, cfg)

                if cfg.use_wandb:
                    wandb.log(logger)

