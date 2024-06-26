"""Adapted from https://github.com/deepmind/annealed_flow_transport."""

from typing import Tuple

import chex
import jax
import jax.numpy as jnp

Array = chex.Array
RandomKey = chex.PRNGKey
Samples = chex.Array

assert_trees_all_equal_shapes = chex.assert_trees_all_equal_shapes


def log_effective_sample_size(log_weights: Array) -> Array:
    """Adapted to set max of ESS to 1 (fraction rather than absolute number of samples).

    Numerically stable computation of log of effective sample size.
    ESS := (sum_i weight_i)^2 / (sum_i weight_i^2) and so working in terms of logs
    log ESS = 2 log sum_i (log exp log weight_i) - log sum_i (exp 2 log weight_i )

    Args:
        log_weights: Array of shape (num_batch). log of normalized weights.
    Returns:
        Scalar log ESS.
    """
    chex.assert_rank(log_weights, 1)
    n_samples = log_weights.shape[0]
    first_term = 2.*jax.scipy.special.logsumexp(log_weights)
    second_term = jax.scipy.special.logsumexp(2.*log_weights)
    chex.assert_equal_shape([first_term, second_term])
    return first_term - second_term - jnp.log(n_samples)


def simple_resampling(key: RandomKey, log_weights: chex.Array, samples: chex.ArrayTree) -> Tuple[Array, chex.ArrayTree]:
    chex.assert_rank(log_weights, 1)
    num_batch = log_weights.shape[0]
    indices = jax.random.categorical(key, log_weights,
                                   shape=(num_batch,))
    take_lambda = lambda x: jnp.take(x, indices, axis=0)
    resamples = jax.tree_util.tree_map(take_lambda, samples)
    assert_trees_all_equal_shapes(resamples, samples)
    return indices, resamples


def smc_simple_resampling(key: RandomKey, log_weights: chex.Array, samples: chex.ArrayTree) -> Tuple[Array, chex.ArrayTree]:
    chex.assert_rank(log_weights, 1)
    num_batch = log_weights.shape[0]
    indices = jax.random.categorical(key, log_weights,
                                   shape=(num_batch,))
    take_lambda = lambda x: jnp.take(x, indices, axis=0)
    resamples = jax.tree_util.tree_map(take_lambda, samples)
    assert_trees_all_equal_shapes(resamples, samples)
    log_w_new = jnp.zeros_like(log_weights) + jax.nn.logsumexp(log_weights) - jnp.log(log_weights.shape[0])
    return resamples, log_w_new


def optionally_resample(key: RandomKey, log_weights: Array, samples: Samples,
                        resample_threshold: float) -> Tuple[Array, Array, Array]:
    """Call simple_resampling on log_weights/samples if ESS is below threshold."""
    # In the case where we don't resample we just return the current
    # samples and weights.
    # lamdba_no_resample will do that on the tuple given to jax.lax.cond below.
    lambda_no_resample = lambda x: (x[2], x[1])
    lambda_resample = lambda x: smc_simple_resampling(*x)
    threshold_sample_size = resample_threshold
    log_ess = log_effective_sample_size(log_weights)
    samples, log_w_new = jax.lax.cond(log_ess < jnp.log(threshold_sample_size), lambda_resample,
                      lambda_no_resample, (key, log_weights, samples))
    return samples, log_w_new, log_ess