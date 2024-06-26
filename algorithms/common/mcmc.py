"""Code builds on https://github.com/google-deepmind/annealed_flow_transport """
import time

from absl import logging
from algorithms.annealed_flow_transport import densities
from algorithms.common import markov_kernel
from algorithms.common import samplers
import algorithms.common.types as tp
import chex
import jax
import jax.numpy as jnp
import numpy as np

Array = tp.Array
LogDensityNoStep = tp.LogDensityNoStep
InitialSampler = tp.InitialSampler
RandomKey = tp.RandomKey
MarkovKernelApply = tp.MarkovKernelApply
assert_equal_shape = chex.assert_equal_shape
AlgoResultsTuple = tp.AlgoResultsTuple
ParticleState = tp.ParticleState


def outer_loop_mcmc(key: RandomKey,
                    num_iters: int,
                    record_expectations,
                    config) -> AlgoResultsTuple:
    """The outer loop for Annealed Flow Transport Monte Carlo.

  Args:
    key: A Jax random key.
    num_iters: Number of iterations of MCMC to run.
    record_expectations: Function for recording values of expectations.
    config: A ConfigDict containing the configuration.
  Returns:
    An AlgoResults tuple containing a summary of the results.
  """
    final_log_density = getattr(densities, config.final_config.density)(
        config.final_config, config.sample_shape[0])
    initial_sampler = getattr(samplers,
                              config.initial_sampler_config.initial_sampler)(
        config.initial_sampler_config)

    num_temps = 2
    key, subkey = jax.random.split(key)

    samples = initial_sampler(subkey, config.batch_size, config.sample_shape)
    log_weights = -jnp.log(config.batch_size) * jnp.ones(config.batch_size)

    dummy_density_by_step = lambda unused_step, x: final_log_density(x)
    final_step = 1

    markov_kernel_dummy_step = markov_kernel.MarkovTransitionKernel(
        config.mcmc_config, dummy_density_by_step, num_temps)

    logging.info('Jitting step...')
    fast_markov_kernel = jax.jit(
        lambda x, y: markov_kernel_dummy_step(final_step, x, y))

    logging.info('Performing initial step redundantly for accurate timing...')
    initial_start_time = time.time()
    fast_markov_kernel(key, samples)
    initial_finish_time = time.time()
    initial_time_diff = initial_finish_time - initial_start_time
    logging.info('Initial step time / seconds  %f: ', initial_time_diff)
    logging.info('Launching training...')
    log_normalizer_estimate = 0.
    start_time = time.time()
    for step in range(num_iters):
        subkey, key = jax.random.split(key)
        samples, acceptance = fast_markov_kernel(subkey, samples)
        acceptance_nuts = float(np.asarray(acceptance[0]))
        acceptance_hmc = float(np.asarray(acceptance[1]))
        particle_state = ParticleState(
            samples=samples,
            log_weights=log_weights,
            log_normalizer_estimate=log_normalizer_estimate)
        record_expectations(particle_state)
        if step % config.mcmc_config.report_step == 0:
            logging.info(
                'Step %05d: Acceptance rate NUTS %f Acceptance rate HMC %f',
                step, acceptance_nuts, acceptance_hmc
            )

    finish_time = time.time()
    delta_time = finish_time - start_time
    logging.info('Delta time / seconds  %f: ', delta_time)
    logging.info('Log normalizer estimate %f: ', log_normalizer_estimate)
    results = AlgoResultsTuple(
        test_samples=samples,
        test_log_weights=log_weights,
        log_normalizer_estimate=log_normalizer_estimate,
        delta_time=delta_time,
        initial_time_diff=initial_time_diff)
    return results
