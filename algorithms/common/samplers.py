"""Code builds on https://github.com/google-deepmind/annealed_flow_transport """
from typing import Tuple

import algorithms.common.types as tp
import jax

RandomKey = tp.RandomKey
Array = tp.Array


class NormalDistribution(object):
    """A wrapper for the univariate normal sampler."""

    def __init__(self, config):
        self._config = config

    def __call__(self,
                 key: RandomKey,
                 num_samples: int,
                 sample_shape: Tuple[int]) -> Array:
        batched_sample_shape = (num_samples,) + sample_shape
        return jax.random.normal(key,
                                 shape=batched_sample_shape)


class MultivariateNormalDistribution(object):
    """A wrapper for the multivariate normal sampler."""

    def __init__(self, config):
        self._config = config

    def __call__(self, key: RandomKey, num_samples: int,
                 sample_shape: Tuple[int]) -> Array:
        batched_sample_shape = (num_samples,) + sample_shape
        return jax.random.normal(key, shape=batched_sample_shape)
