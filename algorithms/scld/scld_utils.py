from flax import traverse_util
from flax.traverse_util import flatten_dict, unflatten_dict

import algorithms.common.types as tp
import chex
import jax
import jax.numpy as jnp
import numpyro.distributions as npdist

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
                 num_temps: int,
                 target_grad_clip=None
                 ):
        self._initial_log_density = initial_log_density
        if target_grad_clip > 0.:
            self._final_log_density = lambda x: jnp.clip(final_log_density(x), -target_grad_clip, target_grad_clip)
        else:
            self._final_log_density = final_log_density
        self._num_temps = num_temps

    def get_beta(self,  # todo add beta schedule?
                 step):
        final_step = self._num_temps - 1
        beta = step / final_step
        return beta

    def __call__(self, step: int, x):
        log_densities_final = self._final_log_density(x)
        log_densities_initial = self._initial_log_density(x)
        beta = self.get_beta(step)
        interpolated_densities = (1. - beta) * log_densities_initial + beta * log_densities_final
        return interpolated_densities


def gradient_step(model_state, grads):
    grads_flat = flatten_dict(grads)
    grads_avg = unflatten_dict(jax.tree_map(lambda g: g.mean(0), grads_flat))
    return model_state.apply_gradients(grads=grads_avg)


def sample_kernel(rng_key, mean, scale):
    eps = jax.random.normal(rng_key, shape=(mean.shape[0],))
    return mean + scale * eps


def log_prob_kernel(x, mean, scale):
    dist = npdist.Independent(npdist.Normal(loc=mean, scale=scale), 1)
    return dist.log_prob(x)


def print_results(step, logger, config):
    if config.verbose:
        try:
            print(f'Step {step}: ELBO w. SMC {logger["metric/smc_ELBO"]}; ΔlnZ {logger["metric/smc_delta_lnZ"]}')
            print(f'Step {step}: ELBO w/o. SMC {logger["metric/model_ELBO"]}; ΔlnZ {logger["metric/model_delta_lnZ"]}')
        except:
            print(f'Step {step}: ELBO w. SMC {logger["metric/smc_ELBO"]}')
            print(f'Step {step}: ELBO w/o. SMC {logger["metric/model_ELBO"]}')


def flattened_traversal(fn):
    def mask(data):
        flat = traverse_util.flatten_dict(data)
        return traverse_util.unflatten_dict({k: fn(k, v) for k, v in flat.items()})

    return mask