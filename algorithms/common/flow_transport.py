"""Code builds on https://github.com/google-deepmind/annealed_flow_transport """
from typing import Any, Tuple, Union

from algorithms.common import resampling
import algorithms.common.types as tp
import chex
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

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


class GeometricAnnealingSchedule(object):
    """Container computing a geometric annealing schedule between log densities."""

    def __init__(self,
                 initial_log_density: LogDensityNoStep,
                 final_log_density: LogDensityNoStep,
                 num_temps: int):
        self._initial_log_density = initial_log_density
        self._final_log_density = final_log_density
        self._num_temps = num_temps

    def get_beta(self,
                 step):
        final_step = self._num_temps - 1
        beta = step / final_step
        return beta

    def __call__(self,
                 step: int,
                 samples: Samples):
        log_densities_final = self._final_log_density(samples)
        log_densities_initial = self._initial_log_density(samples)
        beta = self.get_beta(step)
        interpolated_densities = (
                                         1. - beta) * log_densities_initial + beta * log_densities_final
        return interpolated_densities


def get_delta_no_flow(samples: Samples,
                      log_density: LogDensityByStep,
                      step: int) -> Array:
    log_density_values_current = log_density(step, samples)
    log_density_values_previous = log_density(step - 1, samples)
    assert_equal_shape([log_density_values_current, log_density_values_previous])
    deltas = log_density_values_previous - log_density_values_current
    return deltas


def get_delta(samples: Samples,
              flow_apply: FlowApply,
              flow_params: FlowParams,
              log_density: LogDensityByStep,
              step: int) -> Array:
    """Get density difference between current target and push forward of previous.

  Args:
    samples: Array containing samples of shape (batch,) + sample_shape.
    flow_apply: function that applies the flow.
    flow_params: parameters of the flow.
    log_density: function returning the log_density of a sample at given step.
    step: current step.

  Returns:
    deltas: an array containing the difference for each sample.
  """
    transformed_samples, log_det_jacs = flow_apply(flow_params, samples)
    assert_trees_all_equal_shapes(transformed_samples, samples)
    log_density_values_current = log_density(step, transformed_samples)
    log_density_values_previous = log_density(step - 1, samples)
    assert_equal_shape([log_density_values_current, log_density_values_previous])
    assert_equal_shape([log_density_values_previous, log_det_jacs])
    deltas = log_density_values_previous - log_density_values_current - log_det_jacs
    return deltas


def get_delta_reverse(samples: Samples,
                      inv_flow_apply: FlowApply,
                      flow_params: FlowParams,
                      log_density: LogDensityByStep,
                      step: int) -> Array:
    """Get density difference between current target and push forward of previous.

  Args:
    samples: Array containing samples of shape (batch,) + sample_shape.
    inv_flow_apply: function that applies the flow.
    flow_params: parameters of the flow.
    log_density: function returning the log_density of a sample at given step.
    step: current step.

  Returns:
    deltas: an array containing the difference for each sample.
  """
    transformed_samples, log_det_jacs = inv_flow_apply(flow_params, samples)
    assert_trees_all_equal_shapes(transformed_samples, samples)
    log_density_values_current = log_density(step, samples)
    log_density_values_previous = log_density(step - 1, transformed_samples)
    assert_equal_shape([log_density_values_current, log_density_values_previous])
    assert_equal_shape([log_density_values_previous, log_det_jacs])
    deltas = (log_density_values_current - (log_density_values_previous + log_det_jacs))
    return deltas


def get_delta_path_grad(samples: Samples,
                        flow_apply: FlowApply,
                        inv_flow_apply: FlowApply,
                        flow_params: FlowParams,
                        log_density: LogDensityByStep,
                        step: int) -> Array:
    """Like get_delta above but with gradient changed to use path estimator.

  See https://arxiv.org/abs/1703.09194 and https://arxiv.org/abs/2207.08219

  Args:
    samples: Array containing samples of shape (batch,) + sample_shape.
    flow_apply: function that applies the flow.
    inv_flow_apply: function that applies the inverse flow.
    flow_params: parameters of the flow.
    log_density: function returning the log_density of a sample at given step.
    step: current step.

  Returns:
    deltas: an array containing the difference for each sample.
  """
    transformed_samples, _ = flow_apply(flow_params, samples)
    assert_trees_all_equal_shapes(transformed_samples, samples)
    log_density_target = log_density(step, transformed_samples)

    def variational_density(params, input_samples):
        initial_samples, log_det_jacs = inv_flow_apply(params, input_samples)
        assert_trees_all_equal_shapes(initial_samples, input_samples)
        log_density_base = log_density(step - 1, initial_samples)
        assert_equal_shape([log_density_base, log_det_jacs])
        return log_density_base + log_det_jacs

    log_density_q = variational_density(jax.lax.stop_gradient(flow_params),
                                        transformed_samples)
    assert_equal_shape([log_density_target, log_density_q])
    return log_density_q - log_density_target


def get_batch_parallel_free_energy_increment(samples: Samples,
                                             flow_apply: FlowApply,
                                             flow_params: FlowParams,
                                             log_density: LogDensityByStep,
                                             step: int) -> Array:
    """Get the log normalizer increments in case where there is no resampling.

  Args:
    samples: (num_batch, num_dim)
    flow_apply: Apply the flow.
    flow_params: Parameters of the flow.
    log_density: Value of the log density.
    step: Step of the algorithm.

  Returns:
    Scalar array containing the increments.
  """
    deltas = get_delta(samples, flow_apply, flow_params, log_density, step)
    chex.assert_rank(deltas, 1)
    # The mean takes the average over the batch. This is equivalent to delaying
    # the average until all temperatures have been accumulated.
    return jnp.mean(deltas)


def transport_free_energy_estimator(samples: Samples,
                                    log_weights: Array,
                                    flow_apply: FlowApply,
                                    inv_flow_apply: Union[FlowApply, Any],
                                    flow_params: FlowParams,
                                    log_density: LogDensityByStep,
                                    step: int,
                                    use_path_gradient: bool) -> Array:
    """Compute an estimate of the free energy.

  Args:
    samples: Array representing samples (batch,) + sample_shape
    log_weights: scalar representing sample weights (batch,)
    flow_apply: function that applies the flow.
    inv_flow_apply: function that applies the inverse flow or None.
    flow_params: parameters of the flow.
    log_density:  function returning the log_density of a sample at given step.
    step: current step
    use_path_gradient: Whether or not to modify gradients to use path estimator.

  Returns:
    Estimate of the free_energy.
  """
    if not use_path_gradient:
        deltas = get_delta(samples,
                           flow_apply,
                           flow_params,
                           log_density,
                           step)
    else:
        deltas = get_delta_path_grad(samples,
                                     flow_apply,
                                     inv_flow_apply,
                                     flow_params,
                                     log_density,
                                     step)
    assert_equal_shape([deltas, log_weights])
    return jnp.sum(jax.nn.softmax(log_weights) * deltas)


def get_log_normalizer_increment_no_flow(deltas: Array,
                                         log_weights: Array) -> Array:
    assert_equal_shape([deltas, log_weights])
    normalized_log_weights = jax.nn.log_softmax(log_weights)
    total_terms = normalized_log_weights - deltas
    assert_equal_shape([normalized_log_weights, log_weights, total_terms])
    increment = logsumexp(total_terms)
    elbo_inc = jnp.sum(jnp.exp(normalized_log_weights) * - deltas)
    return increment, elbo_inc


def get_log_normalizer_increment_no_flow_verse(deltas: Array,
                                               log_weights: Array) -> Array:
    assert_equal_shape([deltas, log_weights])
    normalized_log_weights = jax.nn.log_softmax(log_weights)
    total_terms = normalized_log_weights + deltas
    assert_equal_shape([normalized_log_weights, log_weights, total_terms])
    increment = -logsumexp(total_terms)
    elbo_inc = jnp.sum(jnp.exp(normalized_log_weights) * - deltas)
    return increment, elbo_inc


def get_log_normalizer_increment_no_flow_craft(deltas: Array,
                                               log_weights: Array) -> Array:
    assert_equal_shape([deltas, log_weights])
    normalized_log_weights = jax.nn.log_softmax(log_weights)
    total_terms = normalized_log_weights - deltas
    assert_equal_shape([normalized_log_weights, log_weights, total_terms])
    increment = logsumexp(total_terms)
    return increment


def get_log_normalizer_increment(samples: Samples,
                                 log_weights: Array,
                                 flow_apply: FlowApply,
                                 flow_params: FlowParams,
                                 log_density: LogDensityByStep,
                                 step: int,
                                 reverse=False) -> Array:
    """Get the increment in the log of the normalizing constant estimate.

  Args:
    samples: Array representing samples (batch,) + sample_shape
    log_weights: scalar representing sample weights (batch,)
    flow_apply: function that applies the flow.
    flow_params: parameters of the flow.
    log_density:  function returning the log_density of a sample at given step.
    step: current step

  Returns:
    Scalar Array, logarithm of normalizing constant increment.
  """

    if reverse:
        deltas = -get_delta_reverse(samples,
                                    flow_apply,
                                    flow_params,
                                    log_density,
                                    step)
        increment = get_log_normalizer_increment_no_flow_verse(deltas, log_weights)
    else:
        deltas = get_delta(samples,
                           flow_apply,
                           flow_params,
                           log_density,
                           step)
        increment = get_log_normalizer_increment_no_flow(deltas, log_weights)

    return increment


def get_log_normalizer_increment_craft(samples: Samples,
                                       log_weights: Array,
                                       flow_apply: FlowApply,
                                       flow_params: FlowParams,
                                       log_density: LogDensityByStep,
                                       step: int) -> Array:
    """Get the increment in the log of the normalizing constant estimate.

  Args:
    samples: Array representing samples (batch,) + sample_shape
    log_weights: scalar representing sample weights (batch,)
    flow_apply: function that applies the flow.
    flow_params: parameters of the flow.
    log_density:  function returning the log_density of a sample at given step.
    step: current step

  Returns:
    Scalar Array, logarithm of normalizing constant increment.
  """
    deltas = get_delta(samples,
                       flow_apply,
                       flow_params,
                       log_density,
                       step)
    increment = get_log_normalizer_increment_no_flow_craft(deltas, log_weights)
    return increment


def reweight_no_flow(log_weights_old: Array,
                     deltas: Array) -> Array:
    log_weights_new_unorm = log_weights_old - deltas
    log_weights_new = jax.nn.log_softmax(log_weights_new_unorm)
    return log_weights_new


def reweight(log_weights_old: Array,
             samples: Samples,
             flow_apply: FlowApply,
             flow_params: FlowParams,
             log_density: LogDensityByStep,
             step: int,
             reverse=False) -> Array:
    """Compute the new weights from the old ones and the deltas.

  Args:
    log_weights_old: scalar representing previous sample weights (batch,)
    samples: Array representing samples (batch,) + sample_shape
    flow_apply: function that applies the flow.
    flow_params: parameters of the flow.
    log_density:  function returning the log_density of a sample at given step.
    step: current step
  Returns:
    logarithm of new weights.
  """
    if reverse:
        deltas = -get_delta_reverse(samples,
                                    flow_apply,
                                    flow_params,
                                    log_density,
                                    step)

    else:
        deltas = get_delta(samples,
                           flow_apply,
                           flow_params,
                           log_density,
                           step)

    log_weights_new = reweight_no_flow(log_weights_old, deltas)
    return log_weights_new


def update_samples_log_weights(
        flow_apply: FlowApply, markov_kernel_apply: MarkovKernelApply,
        flow_params: FlowParams, samples: Samples, log_weights: Array,
        key: RandomKey, log_density: LogDensityByStep, step: int,
        use_resampling: bool, use_markov: bool,
        resample_threshold: float, reverse=False) -> Tuple[Array, Array, AcceptanceTuple]:
    """Update samples and log weights once the flow has been learnt."""
    transformed_samples, _ = flow_apply(flow_params, samples)
    assert_trees_all_equal_shapes(transformed_samples, samples)
    log_weights_new = reweight(log_weights, samples, flow_apply, flow_params, log_density, step, reverse)
    assert_equal_shape([log_weights_new, log_weights])
    if use_resampling:
        subkey, key = jax.random.split(key)
        resampled_samples, log_weights_resampled = resampling.optionally_resample(
            subkey, log_weights_new, transformed_samples, resample_threshold)
        assert_trees_all_equal_shapes(resampled_samples, transformed_samples)
        assert_equal_shape([log_weights_resampled, log_weights_new])
    else:
        resampled_samples = transformed_samples
        log_weights_resampled = log_weights_new
    if use_markov:
        markov_samples, acceptance_tuple = markov_kernel_apply(
            step - 1 if reverse else step, key, resampled_samples)
    else:
        markov_samples = resampled_samples
        acceptance_tuple = (1., 1., 1.)
    return markov_samples, log_weights_resampled, acceptance_tuple
