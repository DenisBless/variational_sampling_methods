"""
Sequential Monte Carlo (SMC) sampler algorithm
For further details see: Del Moral, Doucet and Jasra. 2006. Sequential Monte Carlo samplers.
Code builds on https://github.com/google-deepmind/annealed_flow_transport
"""

import time
from typing import Tuple

import wandb
from algorithms.common import flow_transport
from algorithms.common import resampling
import algorithms.common.types as tp
import chex
import jax
import jax.numpy as jnp
import numpy as np

from algorithms.common.eval_methods.sis_methods import get_eval_fn
from algorithms.common.eval_methods.utils import extract_last_entry
from targets.base_target import Target
from utils.print_util import print_results

Array = tp.Array
LogDensityNoStep = tp.LogDensityNoStep
InitialSampler = tp.InitialSampler
RandomKey = tp.RandomKey
MarkovKernelApply = tp.MarkovKernelApply
LogDensityByStep = tp.LogDensityByStep
assert_equal_shape = chex.assert_equal_shape
assert_trees_all_equal_shapes = chex.assert_trees_all_equal_shapes
AlgoResultsTuple = tp.AlgoResultsTuple
ParticleState = tp.ParticleState


# max_elbo = -10000000000000

def inner_loop(
        key: RandomKey,
        markov_kernel_apply: MarkovKernelApply,
        samples: Array, log_weights: Array,
        log_density: LogDensityByStep, step: int, alg_cfg, reverse=False
) -> Tuple[Array, Array, Array, Array]:
    """Inner loop of the algorithm.

  Args:
    key: A JAX random key.
    markov_kernel_apply: functional that applies the Markov transition kernel.
    samples: Array containing samples.
    log_weights: Array containing log_weights.
    log_density:  function returning the log_density of a sample at given step.
    step: int giving current step of algorithm.
    config: experiment configuration.

  Returns:
    samples_final: samples after the full inner loop has been performed.
    log_weights_final: log_weights after the full inner loop has been performed.
    log_normalizer_increment: Scalar log of normalizing constant increment.
    Acceptance_rates: Acceptance rates of samplers.
  """

    deltas = flow_transport.get_delta_no_flow(samples, log_density, step)
    if reverse:
        deltas = -deltas
    log_normalizer_increment = flow_transport.get_log_normalizer_increment_no_flow(
        deltas, log_weights)
    log_weights_new = flow_transport.reweight_no_flow(log_weights, deltas)
    if alg_cfg.use_resampling:
        subkey, key = jax.random.split(key)
        resampled_samples, log_weights_resampled = resampling.optionally_resample(
            subkey, log_weights_new, samples, alg_cfg.resample_threshold)
        assert_trees_all_equal_shapes(resampled_samples, samples)
        assert_equal_shape([log_weights_resampled, log_weights_new])
    else:
        resampled_samples = samples
        log_weights_resampled = log_weights_new
    markov_samples, acceptance_tuple = markov_kernel_apply(
        step, key, resampled_samples)

    return markov_samples, log_weights_resampled, log_normalizer_increment, acceptance_tuple


def get_short_inner_loop(markov_kernel_by_step: MarkovKernelApply,
                         density_by_step: LogDensityByStep,
                         config):
    """Get a short version of inner loop."""

    def short_inner_loop(rng_key: RandomKey,
                         loc_samples: Array,
                         loc_log_weights: Array,
                         loc_step: int):
        return inner_loop(rng_key,
                          markov_kernel_by_step,
                          loc_samples,
                          loc_log_weights,
                          density_by_step,
                          loc_step,
                          config)

    return short_inner_loop


def get_short_reverse_inner_loop(markov_kernel_by_step: MarkovKernelApply,
                                 density_by_step: LogDensityByStep,
                                 config):
    """Get a short version of inner loop."""

    def short_inner_loop(rng_key: RandomKey,
                         loc_samples: Array,
                         loc_log_weights: Array,
                         loc_step: int):
        return inner_loop(rng_key,
                          markov_kernel_by_step,
                          loc_samples,
                          loc_log_weights,
                          density_by_step,
                          loc_step,
                          config,
                          reverse=True)

    return short_inner_loop


def outer_loop_smc(density_by_step: LogDensityByStep,
                   initial_sampler: InitialSampler,
                   target: Target,
                   markov_kernel_by_step: MarkovKernelApply,
                   key: RandomKey,
                   cfg) -> AlgoResultsTuple:
    """The outer loop for Annealed Flow Transport Monte Carlo.

  Args:
    density_by_step: The log density for each annealing step.
    initial_sampler: A function that produces the initial samples.
    markov_kernel_by_step: Markov transition kernel for each annealing step.
    key: A Jax random key.
    config: A ConfigDict containing the configuration.

  Returns:
    An AlgoResults tuple containing a summary of the results.
  """
    key, subkey = jax.random.split(key)
    alg_cfg = cfg.algorithm
    num_temps = alg_cfg.num_temps
    target_samples = target.sample(jax.random.PRNGKey(0), (cfg.eval_samples,))

    samples = initial_sampler(seed=jax.random.PRNGKey(0), sample_shape=(alg_cfg.batch_size,))
    log_weights = -jnp.log(alg_cfg.batch_size) * jnp.ones(alg_cfg.batch_size)

    inner_loop_jit = jax.jit(get_short_inner_loop(markov_kernel_by_step, density_by_step, alg_cfg))
    reverse_inner_loop_jit = jax.jit(get_short_reverse_inner_loop(markov_kernel_by_step, density_by_step, alg_cfg))

    eval_fn, logger = get_eval_fn(cfg, target, target_samples)

    ln_z = 0.
    elbo = 0.
    start_time = time.time()

    acceptance_hmc = []
    acceptance_rwm = []

    for step in range(1, num_temps):
        subkey, key = jax.random.split(key)
        samples, log_weights, ln_z_inc, acceptance = inner_loop_jit(
            subkey, samples, log_weights, step)
        ln_z_inc, elbo_inc = ln_z_inc
        acceptance_hmc.append(float(np.asarray(acceptance[0])))
        acceptance_rwm.append(float(np.asarray(acceptance[1])))
        ln_z += ln_z_inc
        elbo += elbo_inc

    finish_time = time.time()
    delta_time = finish_time - start_time

    smc_nfe = 2 * alg_cfg.batch_size * (alg_cfg.num_temps - 1)
    mcmc_nfe = 0.  # alg_cfg.batch_size * (alg_cfg.num_temps - 1) * (
    # alg_cfg.common.hmc_num_leapfrog_steps * alg_cfg.common.hmc_steps_per_iter + alg_cfg.common.rwm_steps_per_iter)
    nfe = smc_nfe + mcmc_nfe

    if cfg.compute_forward_metrics and (target_samples is not None):

        tar_samples = target_samples[:samples.shape[0]]
        log_weights = -jnp.log(alg_cfg.batch_size) * jnp.ones(alg_cfg.batch_size)

        neg_fwd_ln_z = 0.
        neg_eubo = 0.

        for step in range(num_temps, 1, -1):
            subkey, key = jax.random.split(key)
            fwd_samples, log_weights, incs, _ = reverse_inner_loop_jit(subkey, tar_samples, log_weights, step)
            rev_ln_z_inc, eubo_inc = incs
            neg_fwd_ln_z += rev_ln_z_inc
            neg_eubo += eubo_inc

        eubo = -neg_eubo
        fwd_ln_z = - neg_fwd_ln_z

    else:
        fwd_ln_z = None
        eubo = None

    logger = eval_fn(samples, elbo, ln_z, eubo, fwd_ln_z)

    logger["stats/wallclock"] = [delta_time]
    logger["stats/nfe"] = [nfe]
    logger["other/avg_acceptance_hmc"] = [sum(acceptance_hmc) / len(acceptance_hmc)]
    logger["other/avg_acceptance_rwm"] = [sum(acceptance_rwm) / len(acceptance_rwm)]

    print_results(0, logger, cfg)

    if cfg.use_wandb:
        wandb.log(extract_last_entry(logger))
