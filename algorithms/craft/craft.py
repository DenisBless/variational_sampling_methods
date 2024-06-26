"""Code builds on https://github.com/google-deepmind/annealed_flow_transport """
import pickle
from time import time
from typing import Any, Tuple, Union

import wandb
from algorithms.common import flow_transport
import algorithms.common.types as tp
import jax
import jax.numpy as jnp
import optax

from algorithms.common.eval_methods.sis_methods import get_eval_fn
from algorithms.common.eval_methods.utils import extract_last_entry
from algorithms.common.utils import reverse_transition_params
from targets.base_target import Target
from utils.path_utils import project_path
from utils.print_util import print_results

Array = tp.Array
Samples = tp.Samples
OptState = tp.OptState
UpdateFn = tp.UpdateFn
FlowParams = tp.FlowParams
FlowApply = tp.FlowApply
LogDensityNoStep = tp.LogDensityNoStep
InitialSampler = tp.InitialSampler
RandomKey = tp.RandomKey
SamplesTuple = tp.SamplesTuple
FreeEnergyAndGrad = tp.FreeEnergyAndGrad
MarkovKernelApply = tp.MarkovKernelApply
LogDensityByStep = tp.LogDensityByStep
AcceptanceTuple = tp.AcceptanceTuple
LogWeightsTuple = tp.LogWeightsTuple
AlgoResultsTuple = tp.AlgoResultsTuple
ParticleState = tp.ParticleState


def save_model(model_path, params, cfg, step):
    pickle.dump(params, open(project_path(f'{model_path}/{step}.pkl'), 'wb'))


def eval_craft(flow_apply,
               flow_inv_apply,
               density_by_step,
               target,
               markov_kernel_by_step,
               initial_sampler,
               target_samples,
               cfg):
    def short_eval_craft(transition_params, key):
        """A single pass of CRAFT with fixed flows.

      Uses Scan step requiring trees that have the same structure as the base input
      but with each leaf extended with an extra array index of size num_transitions.
      We call this an extended tree.

      Args:
        key: A JAX random key.
        transition_params: Extended tree of flow parameters.
        flow_apply: function that applies the flow.
        markov_kernel_apply: function that applies the Markov transition kernel.
        initial_sampler: A function that produces the initial samples.
        log_density: A function evaluating the log density for each step.
        cfg: A cfgDict containing the cfguration.
      Returns:
        ParticleState containing samples, log_weights, log_normalizer_estimate.
      """
        subkey, key = jax.random.split(key)
        initial_samples = initial_sampler(seed=subkey, sample_shape=(cfg.eval_samples,))
        initial_log_weights = -jnp.log(cfg.eval_samples) * jnp.ones(cfg.eval_samples)

        def scan_step(passed_state, per_step_input):
            samples, log_weights = passed_state

            flow_params, key, inner_step = per_step_input
            log_normalizer_increment = flow_transport.get_log_normalizer_increment(
                samples, log_weights, flow_apply, flow_params, density_by_step, inner_step)
            next_samples, next_log_weights, acceptance_tuple = flow_transport.update_samples_log_weights(
                flow_apply=flow_apply, markov_kernel_apply=markov_kernel_by_step,
                flow_params=flow_params, samples=samples, log_weights=log_weights,
                key=key, log_density=density_by_step, step=inner_step,
                use_resampling=cfg.algorithm.use_resampling, use_markov=cfg.algorithm.use_markov,
                resample_threshold=cfg.algorithm.resample_threshold)
            next_passed_state = (next_samples, next_log_weights)
            per_step_output = (log_normalizer_increment, acceptance_tuple)
            return next_passed_state, per_step_output

        def scan_step_reverse(passed_state, per_step_input):
            samples, log_weights = passed_state
            flow_params, key, inner_step = per_step_input
            log_normalizer_increment = flow_transport.get_log_normalizer_increment(
                samples, log_weights, flow_inv_apply, flow_params, density_by_step, inner_step, reverse=True)
            next_samples, next_log_weights, acceptance_tuple = flow_transport.update_samples_log_weights(
                flow_apply=flow_inv_apply, markov_kernel_apply=markov_kernel_by_step,
                flow_params=flow_params, samples=samples, log_weights=log_weights,
                key=key, log_density=density_by_step, step=inner_step,
                use_resampling=cfg.algorithm.use_resampling, use_markov=cfg.algorithm.use_markov,
                resample_threshold=cfg.algorithm.resample_threshold, reverse=True)
            next_passed_state = (next_samples, next_log_weights)
            per_step_output = (log_normalizer_increment, acceptance_tuple)
            return next_passed_state, per_step_output

        initial_state = (initial_samples, initial_log_weights)
        inner_steps = jnp.arange(1, cfg.algorithm.num_temps)
        key, sub_key = jax.random.split(key)
        keys = jax.random.split(key, cfg.algorithm.num_temps - 1)
        per_step_inputs = (transition_params, keys, inner_steps)
        final_state, per_step_outputs = jax.lax.scan(scan_step, initial_state, per_step_inputs)
        samples, log_is_weights = final_state

        is_weights = jnp.exp(log_is_weights)
        (log_normalizer_increments, elbo_increments), unused_acceptance_tuples = per_step_outputs
        ln_z = jnp.sum(log_normalizer_increments)
        elbo = jnp.sum(elbo_increments)

        if cfg.compute_forward_metrics and (target_samples is not None):
            initial_samples_reverse = target_samples
            initial_log_weights_reverse = -jnp.log(cfg.eval_samples) * jnp.ones(cfg.eval_samples)
            initial_state_reverse = (initial_samples_reverse, initial_log_weights_reverse)
            inner_steps_reverse = jnp.arange(1, cfg.algorithm.num_temps)[::-1]
            keys_reverse = jax.random.split(sub_key, cfg.algorithm.num_temps - 1)
            per_step_inputs_reverse = (reverse_transition_params(transition_params), keys_reverse, inner_steps_reverse)
            final_state_reverse, per_step_outputs_reverse = jax.lax.scan(scan_step_reverse, initial_state_reverse,
                                                                         per_step_inputs_reverse)
            samples_reverse, log_is_weights_reverse = final_state_reverse
            (log_normalizer_increments_reverse, elbo_increments_reverse), _ = per_step_outputs_reverse

            fwd_ln_z = jnp.sum(log_normalizer_increments_reverse)
            eubo = jnp.sum(elbo_increments_reverse)

            return samples, elbo, ln_z, eubo, fwd_ln_z

        return samples, elbo, ln_z, None, None

    return short_eval_craft


def inner_step_craft(
        key: RandomKey, free_energy_and_grad: FreeEnergyAndGrad,
        flow_params: FlowParams,
        flow_apply: FlowApply, markov_kernel_apply: MarkovKernelApply,
        samples: Array, log_weights: Array, log_density: LogDensityByStep,
        step: int, cfg
) -> Tuple[FlowParams, Array, Array, Samples, Array, AcceptanceTuple]:
    """A temperature step of CRAFT.

  Args:
    key: A JAX random key.
    free_energy_and_grad: function giving estimate of free energy and gradient.
    flow_params: parameters of the flow.
    flow_apply: function that applies the flow.
    markov_kernel_apply: function that applies the Markov transition kernel.
    samples: input samples.
    log_weights: Array containing train/validation/test log_weights.
    log_density:  function returning the log_density of a sample at given step.
    step: int giving current step of algorithm.
    cfg: experiment cfguration.

  Returns:
    flow_grads: Gradient with respect to parameters of flow.
    vfe: Value of the objective for this temperature.
    log_normalizer_increment: Scalar log of normalizing constant increment.
    next_samples: samples after temperature step has been performed.
    new_log_weights: log_weights after temperature step has been performed.
    acceptance_tuple: Acceptance rate of the Markov kernels used.
  """
    vfe, flow_grads = free_energy_and_grad(flow_params,
                                           samples,
                                           log_weights,
                                           step)
    log_normalizer_increment = flow_transport.get_log_normalizer_increment_craft(
        samples, log_weights, flow_apply, flow_params, log_density, step)
    next_samples, next_log_weights, acceptance_tuple = flow_transport.update_samples_log_weights(
        flow_apply=flow_apply, markov_kernel_apply=markov_kernel_apply,
        flow_params=flow_params, samples=samples, log_weights=log_weights,
        key=key, log_density=log_density, step=step,
        use_resampling=cfg.algorithm.use_resampling, use_markov=cfg.algorithm.use_markov,
        resample_threshold=cfg.algorithm.resample_threshold)

    return flow_grads, vfe, log_normalizer_increment, next_samples, next_log_weights, acceptance_tuple


def inner_loop_craft(key: RandomKey, free_energy_and_grad: FreeEnergyAndGrad,
                     opt_update: UpdateFn, opt_states: OptState,
                     transition_params: FlowParams, flow_apply: FlowApply,
                     markov_kernel_apply: MarkovKernelApply,
                     initial_sampler: InitialSampler,
                     log_density: LogDensityByStep, cfg,
                     axis_name=None):
    """Inner loop of CRAFT training.

  Uses Scan step requiring trees that have the same structure as the base input
  but with each leaf extended with an extra array index of size num_transitions.
  We call this an extended tree.

  Args:
    key: A JAX random key.
    free_energy_and_grad: function giving estimate of free energy and gradient.
    opt_update: function that updates the state of flow based on gradients etc.
    opt_states: Extended tree of optimizer states.
    transition_params: Extended tree of flow parameters.
    flow_apply: function that applies the flow.
    markov_kernel_apply: function that applies the Markov transition kernel.
    initial_sampler: A function that produces the initial samples.
    log_density: A function evaluating the log density for each step.
    cfg: A cfgDict containing the cfguration.
    axis_name: None or string for gradient sync when using pmap only.
  Returns:
    final_samples: final samples.
    final_log_weights: Array of final log_weights.
    final_transition_params: Extended tree of updated flow params.
    final_opt_states: Extended tree of updated optimizer parameters.
    overall_free_energy: Total variational free energy.
    log_normalizer_estimate: Estimate of the log normalizers.
  """
    subkey, key = jax.random.split(key)
    initial_samples = initial_sampler(seed=subkey, sample_shape=(cfg.algorithm.batch_size,))

    initial_log_weights = -jnp.log(cfg.algorithm.batch_size) * jnp.ones(
        cfg.algorithm.batch_size)

    def scan_step(passed_state, per_step_input):
        samples, log_weights = passed_state
        flow_params, key, inner_step = per_step_input
        flow_grads, vfe, log_normalizer_increment, next_samples, next_log_weights, acceptance_tuple = inner_step_craft(
            key=key,
            free_energy_and_grad=free_energy_and_grad,
            flow_params=flow_params,
            flow_apply=flow_apply,
            markov_kernel_apply=markov_kernel_apply,
            samples=samples,
            log_weights=log_weights,
            log_density=log_density,
            step=inner_step,
            cfg=cfg)
        next_passed_state = (next_samples, next_log_weights)
        per_step_output = (flow_grads, vfe,
                           log_normalizer_increment, acceptance_tuple)
        return next_passed_state, per_step_output

    initial_state = (initial_samples, initial_log_weights)
    inner_steps = jnp.arange(1, cfg.algorithm.num_temps)
    keys = jax.random.split(key, cfg.algorithm.num_temps - 1)
    per_step_inputs = (transition_params, keys, inner_steps)
    final_state, per_step_outputs = jax.lax.scan(scan_step, initial_state,
                                                 per_step_inputs)
    final_samples, final_log_weights = final_state
    flow_grads, free_energies, log_normalizer_increments, unused_acceptance_tuples = per_step_outputs

    if axis_name:
        flow_grads = jax.lax.pmean(flow_grads, axis_name=axis_name)

    def per_step_update(input_tuple):
        (step_grad, step_opt, step_params) = input_tuple
        step_updates, new_opt_state = opt_update(step_grad,
                                                 step_opt)
        new_step_params = optax.apply_updates(step_params,
                                              step_updates)
        return new_step_params, new_opt_state

    final_transition_params, final_opt_states = jax.lax.map(
        per_step_update, (flow_grads, opt_states, transition_params))

    overall_free_energy = jnp.sum(free_energies)
    log_normalizer_estimate = jnp.sum(log_normalizer_increments)
    return final_samples, final_log_weights, final_transition_params, final_opt_states, overall_free_energy, log_normalizer_estimate


def outer_loop_craft(opt_update: UpdateFn,
                     opt_init_state: OptState,
                     flow_init_params: FlowParams,
                     flow_apply: FlowApply,
                     flow_inv_apply: Union[FlowApply, Any],
                     density_by_step: LogDensityByStep,
                     target: Target,
                     markov_kernel_by_step: MarkovKernelApply,
                     initial_sampler: InitialSampler,
                     key: RandomKey,
                     cfg):
    """Outer loop for CRAFT training.

  Args:
    opt_update: function that updates the state of flow based on gradients etc.
    opt_init_state: initial state variables of the optimizer.
    flow_init_params: initial state of the flow.
    flow_apply: function that applies the flow.
    flow_inv_apply: function that applies the inverse flow or None.
    density_by_step: The log density for different annealing temperatures.
    markov_kernel_by_step: Markov kernel for different annealing temperatures.
    initial_sampler: A function that produces the initial samples.
    key: A Jax random key.
    cfg: A cfgDict containing the cfguration.
    log_step_output: Callable that logs the step output.
    save_checkpoint: None or function that takes params and saves them.
  Returns:
    An AlgoResults tuple containing a summary of the results.
  """

    def free_energy_short(flow_params: FlowParams,
                          samples: Samples,
                          log_weights: Array,
                          step: int) -> Array:
        return flow_transport.transport_free_energy_estimator(
            samples, log_weights, flow_apply, flow_inv_apply, flow_params,
            density_by_step, step, cfg.algorithm.use_path_gradient)

    free_energy_and_grad = jax.value_and_grad(free_energy_short)

    def short_inner_loop(rng_key: RandomKey,
                         curr_opt_states: OptState,
                         curr_transition_params):
        return inner_loop_craft(key=rng_key,
                                free_energy_and_grad=free_energy_and_grad,
                                opt_update=opt_update,
                                opt_states=curr_opt_states,
                                transition_params=curr_transition_params,
                                flow_apply=flow_apply,
                                markov_kernel_apply=markov_kernel_by_step,
                                initial_sampler=initial_sampler,
                                log_density=density_by_step,
                                cfg=cfg)

    inner_loop_jit = jax.jit(short_inner_loop)
    alg_cfg = cfg.algorithm
    num_temps = alg_cfg.num_temps
    eval_freq = max(alg_cfg.iters // cfg.n_evals, 1)
    target_samples = target.sample(jax.random.PRNGKey(0), (cfg.eval_samples,))

    eval_fn, logger = get_eval_fn(cfg, target, target_samples)

    get_metrics = eval_craft(flow_apply, flow_inv_apply, density_by_step, target, markov_kernel_by_step,
                             initial_sampler,
                             target_samples, cfg)

    repeater = lambda x: jnp.repeat(x[None], num_temps - 1, axis=0)
    opt_states = jax.tree_util.tree_map(repeater, opt_init_state)
    transition_params = jax.tree_util.tree_map(repeater, flow_init_params)

    flow_nfe = 0.  # 2 * cfg.algorithm.batch_size * (cfg.algorithm.num_temps - 1)
    mcmc_nfe = 0.  # cfg.algorithm.batch_size * (cfg.algorithm.num_temps - 1) * (
    #         cfg.mcmc_cfg.hmc_num_leapfrog_steps * cfg.mcmc_cfg.hmc_steps_per_iter + cfg.mcmc_cfg.rwm_steps_per_iter)

    logger = {}
    test_elbos = []

    timer = 0
    for step in range(alg_cfg.iters):
        iter_time = time()
        with jax.profiler.StepTraceAnnotation('train', step_num=step):
            key, subkey = jax.random.split(key)
            final_samples, final_log_weights, transition_params, opt_states, overall_free_energy, log_normalizer_estimate = inner_loop_jit(
                subkey, opt_states, transition_params)
            timer += time() - iter_time

            if step % eval_freq == 0:
                key, subkey = jax.random.split(key)
                logger = eval_fn(*get_metrics(transition_params, subkey))
                logger["stats/step"] = [step]
                logger["stats/wallclock"] = [timer]
                logger["stats/nfe"] = [step * (flow_nfe + mcmc_nfe)]

                print_results(step, logger, cfg)

                if cfg.use_wandb:
                    wandb.log(extract_last_entry(logger))

    return test_elbos, logger
