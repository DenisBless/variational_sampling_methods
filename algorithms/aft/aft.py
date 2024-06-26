"""Code builds on https://github.com/google-deepmind/annealed_flow_transport """
import time
from typing import NamedTuple, Tuple, Union

import wandb

from algorithms.common import flow_transport
import algorithms.common.types as tp
import jax
import jax.numpy as jnp
import numpy as np
import optax

from algorithms.common.eval_methods.sis_methods import get_eval_fn
from algorithms.common.eval_methods.utils import extract_last_entry
from targets.base_target import Target
from utils.print_util import print_results

Array = tp.Array
UpdateFn = tp.UpdateFn
OptState = tp.OptState
FlowParams = tp.FlowParams
FlowApply = tp.FlowApply
LogDensityNoStep = tp.LogDensityNoStep
InitialSampler = tp.InitialSampler
RandomKey = tp.RandomKey
SamplesTuple = tp.SamplesTuple
FreeEnergyAndGrad = tp.FreeEnergyAndGrad
MarkovKernelApply = tp.MarkovKernelApply
FreeEnergyEval = tp.FreeEnergyEval
VfesTuple = tp.VfesTuple
LogDensityByStep = tp.LogDensityByStep
AcceptanceTuple = tp.AcceptanceTuple
LogWeightsTuple = tp.LogWeightsTuple
AlgoResultsTuple = tp.AlgoResultsTuple


def get_initial_samples_log_weight_tuples(
        initial_sampler: InitialSampler, key: RandomKey,
        cfg) -> Tuple[SamplesTuple, LogWeightsTuple]:
    """Get initial train/validation/test state depending on cfg."""
    batch_sizes = (cfg.algorithm.batch_size,
                   cfg.algorithm.batch_size,
                   cfg.algorithm.batch_size)
    subkeys = jax.random.split(key, 3)
    samples_tuple = SamplesTuple(*[
        initial_sampler(seed=elem, sample_shape=(batch,))
        for elem, batch in zip(subkeys, batch_sizes)
    ])
    log_weights_tuple = LogWeightsTuple(*[-jnp.log(batch) * jnp.ones(
        batch) for batch in batch_sizes])
    return samples_tuple, log_weights_tuple


def update_tuples(
        samples_tuple: SamplesTuple, log_weights_tuple: LogWeightsTuple,
        key: RandomKey, flow_apply: FlowApply, flow_params: FlowParams,
        markov_kernel_apply: MarkovKernelApply, log_density: LogDensityByStep,
        step: int, cfg) -> Tuple[SamplesTuple, LogWeightsTuple, AcceptanceTuple]:
    """Update the samples and log weights and return diagnostics."""
    samples_list = []
    log_weights_list = []
    acceptance_tuple_list = []
    subkeys = jax.random.split(key, 3)
    for curr_samples, curr_log_weights, subkey in zip(samples_tuple,
                                                      log_weights_tuple,
                                                      subkeys):
        new_samples, new_log_weights, acceptance_tuple = flow_transport.update_samples_log_weights(
            flow_apply=flow_apply,
            markov_kernel_apply=markov_kernel_apply,
            flow_params=flow_params,
            samples=curr_samples,
            log_weights=curr_log_weights,
            key=subkey,
            log_density=log_density,
            step=step,
            use_resampling=cfg.algorithm.use_resampling,
            use_markov=cfg.algorithm.use_markov,
            resample_threshold=cfg.algorithm.resample_threshold)
        samples_list.append(new_samples)
        log_weights_list.append(new_log_weights)
        acceptance_tuple_list.append(acceptance_tuple)
    samples_tuple = SamplesTuple(*samples_list)
    log_weights_tuple = LogWeightsTuple(*log_weights_list)
    test_acceptance_tuple = acceptance_tuple_list[-1]
    return samples_tuple, log_weights_tuple, test_acceptance_tuple


class OptimizationLoopState(NamedTuple):
    opt_state: OptState
    flow_params: FlowParams
    inner_step: int
    opt_vfes: VfesTuple
    best_params: FlowParams
    best_validation_vfe: Array
    best_index: int


def flow_estimate_step(loop_state: OptimizationLoopState,
                       free_energy_and_grad: FreeEnergyAndGrad,
                       train_samples: Array, train_log_weights: Array,
                       outer_step: int, validation_samples: Array,
                       validation_log_weights: Array,
                       free_energy_eval: FreeEnergyEval,
                       opt_update: UpdateFn) -> OptimizationLoopState:
    """A single step of the flow estimation loop."""
    # Evaluate the flow on train and validation particles.
    train_vfe, flow_grads = free_energy_and_grad(loop_state.flow_params,
                                                 train_samples,
                                                 train_log_weights,
                                                 outer_step)
    validation_vfe = free_energy_eval(loop_state.flow_params,
                                      validation_samples,
                                      validation_log_weights,
                                      outer_step)

    # Update the best parameters, best validation vfe and index
    # if the measured validation vfe is better.
    validation_vfe_is_better = validation_vfe < loop_state.best_validation_vfe
    new_best_params = jax.lax.cond(validation_vfe_is_better,
                                   lambda _: loop_state.flow_params,
                                   lambda _: loop_state.best_params,
                                   operand=None)
    new_best_validation_vfe = jnp.where(validation_vfe_is_better,
                                        validation_vfe,
                                        loop_state.best_validation_vfe)
    new_best_index = jnp.where(validation_vfe_is_better,
                               loop_state.inner_step,
                               loop_state.best_index)

    # Update the logs of train and validation vfes.
    new_train_vfes = loop_state.opt_vfes.train_vfes.at[loop_state.inner_step].set(
        train_vfe)
    new_validation_vfes = loop_state.opt_vfes.validation_vfes.at[
        loop_state.inner_step].set(validation_vfe)

    new_opt_vfes = VfesTuple(train_vfes=new_train_vfes,
                             validation_vfes=new_validation_vfes)

    # Apply gradients ready for next round of flow evaluations in the next step.
    updates, new_opt_state = opt_update(flow_grads,
                                        loop_state.opt_state)
    new_flow_params = optax.apply_updates(loop_state.flow_params,
                                          updates)
    new_inner_step = loop_state.inner_step + 1

    # Pack everything into the next loop state.
    new_state_tuple = OptimizationLoopState(new_opt_state, new_flow_params,
                                            new_inner_step, new_opt_vfes,
                                            new_best_params,
                                            new_best_validation_vfe,
                                            new_best_index)
    return new_state_tuple


def flow_estimation_should_continue(loop_state: OptimizationLoopState,
                                    opt_iters: int,
                                    stopping_criterion: str) -> bool:
    """Based on stopping criterion control termination of flow estimation."""
    if stopping_criterion == 'time':
        return loop_state.inner_step < opt_iters
    elif stopping_criterion == 'greedy_time':
        index = loop_state.inner_step
        best_index = loop_state.best_index
        return jnp.logical_and(best_index == index - 1, index < opt_iters)
    else:
        raise NotImplementedError


def optimize_free_energy(
        opt_update: UpdateFn, opt_init_state: OptState,
        flow_init_params: FlowParams, free_energy_and_grad: FreeEnergyAndGrad,
        free_energy_eval: FreeEnergyEval, train_samples: Array,
        train_log_weights: Array, validation_samples: Array,
        validation_log_weights: Array, outer_step: int, opt_iters: int,
        stopping_criterion: str) -> Tuple[FlowParams, VfesTuple]:
    """Optimize an estimate of the free energy.

  Args:
    opt_update: function that updates the state of flow based on gradients etc.
    opt_init_state: initial state variables of the optimizer.
    flow_init_params: initial parameters of the flow.
    free_energy_and_grad: function giving estimate of free energy and gradient.
    free_energy_eval: function giving estimate of free energy only.
    train_samples: Array of shape (batch,)+sample_shape
    train_log_weights: Array of shape (batch,)
    validation_samples: Array of shape (batch,)
    validation_log_weights: Array of shape (batch,)
    outer_step: int giving current outer step of algorithm.
    opt_iters: number of flow estimation iters.
    stopping_criterion: One of 'time' or 'greedy-time'.

  Returns:
    flow_params: optimized flow parameters.
    free_energies: array containing all estimates of free energy.
  """
    opt_state = opt_init_state
    flow_params = flow_init_params
    train_vfes = jnp.zeros(opt_iters)
    validation_vfes = jnp.zeros(opt_iters)
    opt_vfes = VfesTuple(train_vfes, validation_vfes)

    def body_fun(loop_state: OptimizationLoopState) -> OptimizationLoopState:
        return flow_estimate_step(loop_state, free_energy_and_grad, train_samples,
                                  train_log_weights, outer_step, validation_samples,
                                  validation_log_weights, free_energy_eval,
                                  opt_update)

    def cond_fun(loop_state: OptimizationLoopState) -> bool:
        return flow_estimation_should_continue(loop_state, opt_iters,
                                               stopping_criterion)

    initial_loop_state = OptimizationLoopState(opt_state, flow_params, 0,
                                               opt_vfes, flow_params, jnp.inf, -1)
    final_loop_state = jax.lax.while_loop(cond_fun,
                                          body_fun,
                                          initial_loop_state)
    return final_loop_state.best_params, final_loop_state.opt_vfes


def inner_loop(
        key: RandomKey, free_energy_and_grad: FreeEnergyAndGrad,
        free_energy_eval: FreeEnergyEval, opt_update: UpdateFn,
        opt_init_state: OptState, flow_init_params: FlowParams,
        flow_apply: FlowApply, markov_kernel_apply: MarkovKernelApply,
        samples_tuple: SamplesTuple, log_weights_tuple: LogWeightsTuple,
        log_density: LogDensityByStep, step: int, cfg
) -> Tuple[FlowParams, OptState, VfesTuple, Array, AcceptanceTuple, FlowParams]:
    """Inner loop of the algorithm.

  Args:
    key: A JAX random key.
    free_energy_and_grad: function giving estimate of free energy and gradient.
    free_energy_eval: function giving estimate of free energy only.
    opt_update: function that updates the state of flow based on gradients etc.
    opt_init_state: initial state variables of the optimizer.
    flow_init_params: initial parameters of the flow.
    flow_apply: function that applies the flow.
    markov_kernel_apply: functional that applies the Markov transition kernel.
    samples_tuple: Tuple containing train/validation/test samples.
    log_weights_tuple: Tuple containing train/validation/test log_weights.
    log_density:  function returning the log_density of a sample at given step.
    step: int giving current step of algorithm.
    cfg: experiment cfguration.

  Returns:
    samples_final: samples after the full inner loop has been performed.
    log_weights_final: log_weights after the full inner loop has been performed.
    free_energies: array containing all estimates of free energy.
    log_normalizer_increment: Scalar log of normalizing constant increment.
  """
    flow_params, vfes_tuple = optimize_free_energy(
        opt_update=opt_update,
        opt_init_state=opt_init_state,
        flow_init_params=flow_init_params,
        free_energy_and_grad=free_energy_and_grad,
        free_energy_eval=free_energy_eval,
        train_samples=samples_tuple.train_samples,
        train_log_weights=log_weights_tuple.train_log_weights,
        validation_samples=samples_tuple.validation_samples,
        validation_log_weights=log_weights_tuple.validation_log_weights,
        outer_step=step,
        opt_iters=cfg.algorithm.free_energy_iters,
        stopping_criterion=cfg.algorithm.stopping_criterion)
    log_normalizer_increment = flow_transport.get_log_normalizer_increment(
        samples_tuple.test_samples, log_weights_tuple.test_log_weights,
        flow_apply, flow_params, log_density, step)

    samples_tuple, log_weights_tuple, test_acceptance_tuple = update_tuples(
        samples_tuple=samples_tuple,
        log_weights_tuple=log_weights_tuple,
        key=key,
        flow_apply=flow_apply,
        flow_params=flow_params,
        markov_kernel_apply=markov_kernel_apply,
        log_density=log_density,
        step=step,
        cfg=cfg)

    return samples_tuple, log_weights_tuple, vfes_tuple, log_normalizer_increment, test_acceptance_tuple, flow_params


def outer_loop_aft(opt_update: UpdateFn,
                   opt_init_state: OptState,
                   flow_init_params: FlowParams,
                   flow_apply: FlowApply,
                   flow_inv_apply: FlowApply,
                   density_by_step: LogDensityByStep,
                   markov_kernel_by_step: MarkovKernelApply,
                   initial_sampler: InitialSampler,
                   key: RandomKey,
                   target: Target,
                   cfg):
    """The outer loop for Annealed Flow Transport Monte Carlo.

  Args:
    opt_update: A Optax optimizer update function.
    opt_init_state: Optax initial state.
    flow_init_params: Initial parameters for the flow.
    flow_apply: Function that evaluates flow on parameters and samples.
    density_by_step: The log density for different annealing temperatures.
    markov_kernel_by_step: Markov kernel for different annealing temperatures.
    initial_sampler: A function that produces the initial samples.
    key: A Jax random key.
    cfg: A cfgDict containing the cfguration.
    log_step_output: Function to log step output or None.
  Returns:
    An AlgoResults tuple containing a summary of the results.
  """
    alg_cfg = cfg.algorithm
    num_temps = alg_cfg.num_temps

    def free_energy_short(flow_params: FlowParams,
                          samples: Array,
                          log_weights: Array,
                          step: int) -> Array:
        return flow_transport.transport_free_energy_estimator(
            samples, log_weights, flow_apply, None, flow_params, density_by_step,
            step, False)

    free_energy_eval = jax.jit(free_energy_short)
    free_energy_and_grad = jax.value_and_grad(free_energy_short)
    key, subkey = jax.random.split(key)
    target_samples = target.sample(jax.random.PRNGKey(0), (cfg.eval_samples,))

    samples_tuple, log_weights_tuple = get_initial_samples_log_weight_tuples(
        initial_sampler, subkey, cfg)

    eval_fn, logger = get_eval_fn(cfg, target, target_samples)

    def short_inner_loop(rng_key: RandomKey,
                         loc_samples_tuple: SamplesTuple,
                         loc_log_weights_tuple: LogWeightsTuple,
                         loc_step: int):
        return inner_loop(key=rng_key,
                          free_energy_and_grad=free_energy_and_grad,
                          free_energy_eval=free_energy_eval,
                          opt_update=opt_update,
                          opt_init_state=opt_init_state,
                          flow_init_params=flow_init_params,
                          flow_apply=flow_apply,
                          markov_kernel_apply=markov_kernel_by_step,
                          samples_tuple=loc_samples_tuple,
                          log_weights_tuple=loc_log_weights_tuple,
                          log_density=density_by_step,
                          step=loc_step,
                          cfg=cfg)

    def short_reverse_inner_loop(rng_key: RandomKey,
                                 flow_params: FlowParams,
                                 samples: SamplesTuple,
                                 log_weights: LogWeightsTuple,
                                 loc_step: int):
        return reverse_is_inner_loop(
            key=rng_key,
            flow_params=flow_params,
            reverse_flow_apply=flow_inv_apply,
            markov_kernel_apply=markov_kernel_by_step,
            target_samples=samples,
            log_weights=log_weights,
            log_density_by_step=density_by_step,
            step=loc_step,
            cfg=cfg
        )

    inner_loop_jit = jax.jit(short_inner_loop)
    reverse_inner_loop_jit = jax.jit(short_reverse_inner_loop)

    ln_z = 0.
    elbo = 0.
    start_time = time.time()
    logger = {}

    flow_params_all = []

    for step in range(1, num_temps):
        subkey, key = jax.random.split(key)
        samples_tuple, log_weights_tuple, vfes_tuple, incs, test_acceptance, flow_params = inner_loop_jit(
            subkey, samples_tuple, log_weights_tuple, step)
        flow_params_all.append(flow_params)
        acceptance_hmc = float(np.asarray(test_acceptance[0]))
        acceptance_rwm = float(np.asarray(test_acceptance[1]))
        incs, elbo_inc = incs
        ln_z += incs
        elbo += elbo_inc

    finish_time = time.time()
    delta_time = finish_time - start_time

    is_weights = jnp.exp(log_weights_tuple.test_log_weights)
    smc_nfe = 0.  # 2 * cfg.algorithm.batch_size * cfg.algorithm.num_temps
    mcmc_nfe = 0.  # cfg.algorithm.batch_size * cfg.algorithm.num_temps * cfg.mcmc_cfg.hmc_num_leapfrog_steps * \
    # cfg.mcmc_cfg.hmc_steps_per_iter
    nfe = smc_nfe + mcmc_nfe

    if cfg.compute_forward_metrics and (target_samples is not None):

        samples = target_samples
        log_weights = -jnp.log(alg_cfg.batch_size) * jnp.ones(samples.shape[0])

        fwd_ln_z = 0.
        eubo = 0.

        for step in range(num_temps - 1, 0, -1):
            subkey, key = jax.random.split(key)
            samples, log_weights, incs, _ = reverse_inner_loop_jit(subkey, flow_params_all[step - 1], samples,
                                                                   log_weights, step)
            rev_ln_z_inc, eubo_inc = incs
            fwd_ln_z += rev_ln_z_inc
            eubo += eubo_inc

    else:
        fwd_ln_z = None
        eubo = None

    logger = eval_fn(samples_tuple.test_samples, elbo, ln_z, eubo, fwd_ln_z)

    logger["stats/wallclock"] = [delta_time]
    logger["stats/nfe"] = [nfe]

    print_results(0, logger, cfg)

    if cfg.use_wandb:
        wandb.log(extract_last_entry(logger))


def reverse_is_inner_loop(
        key: RandomKey,
        flow_params: FlowParams,
        reverse_flow_apply: FlowApply,
        markov_kernel_apply: MarkovKernelApply,
        target_samples,
        log_weights,
        log_density_by_step: LogDensityByStep,
        step: int,
        cfg
):
    log_normalizer_increment = flow_transport.get_log_normalizer_increment(
        target_samples, log_weights,
        reverse_flow_apply, flow_params, log_density_by_step, step, reverse=True)

    new_samples, new_log_weights, acceptance_tuple = flow_transport.update_samples_log_weights(
        flow_apply=reverse_flow_apply,
        markov_kernel_apply=markov_kernel_apply,
        flow_params=flow_params,
        samples=target_samples,
        log_weights=log_weights,
        key=key,
        log_density=log_density_by_step,
        step=step,
        use_resampling=cfg.algorithm.use_resampling,
        use_markov=cfg.algorithm.use_markov,
        resample_threshold=cfg.algorithm.resample_threshold,
        reverse=True)

    return new_samples, new_log_weights, log_normalizer_increment, acceptance_tuple
