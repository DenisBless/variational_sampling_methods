from typing import List

import chex
import matplotlib.pyplot as plt
import numpyro
import jax
from jax._src.flatten_util import ravel_pytree

from targets.base_target import Target
import numpyro.distributions as pydist
import jax.numpy as jnp

data = {"R": [10, 23, 23, 26, 17, 5, 53, 55, 32, 46, 10, 8, 10, 8, 23, 0, 3, 22, 15, 32, 3],
        "N": [39, 62, 81, 51, 39, 6, 74, 72, 51, 79, 13, 16, 30, 28, 45, 4, 12, 41, 30, 51, 7.],
        "X1": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "X2": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        "tot": 21}

R = jnp.array(data['R'])
N = jnp.array(data['N'])
X1 = jnp.array(data['X1'])
X2 = jnp.array(data['X2'])
tot = data['tot']


def load_model_seeds():
    def model(r):
        tau = numpyro.sample('tau', pydist.Gamma(0.01, 0.01))
        a_0 = numpyro.sample('a_0', pydist.Normal(0, 10))
        a_1 = numpyro.sample('a_1', pydist.Normal(0, 10))
        a_2 = numpyro.sample('a_2', pydist.Normal(0, 10))
        a_12 = numpyro.sample('a_12', pydist.Normal(0, 10))
        with numpyro.plate('J', tot):
            b = numpyro.sample('b', pydist.Normal(0, 1 / jnp.sqrt(tau)))
            logits = a_0 + a_1 * X1 + a_2 * X2 + a_12 * X1 * X2 + b
            r = numpyro.sample('r', pydist.BinomialLogits(logits, N), obs=R)

    model_args = (R,)
    return model, model_args


class Seeds(Target):
    def __init__(self, dim=26, log_Z=None, can_sample=False, sample_bounds=None) -> None:
        super().__init__(dim=dim, log_Z=log_Z, can_sample=can_sample)
        self.data_ndim = dim
        model, model_args = load_model_seeds()
        rng_key = jax.random.PRNGKey(1)
        model_param_info, potential_fn, constrain_fn, _ = numpyro.infer.util.initialize_model(rng_key, model,
                                                                                              model_args=model_args)

        params_flat, unflattener = ravel_pytree(model_param_info[0])
        self.log_prob_model = lambda z: -1. * potential_fn(unflattener(z))

    def get_dim(self):
        return self.dim

    def log_prob(self, x: chex.Array):
        batched = x.ndim == 2

        if not batched:
            x = x[None,]

        # log prob model can only handle unbatched input
        log_probs = jax.vmap(self.log_prob_model)(x)

        if not batched:
            log_probs = jnp.squeeze(log_probs, axis=0)

        return log_probs

    def visualise(self, samples: chex.Array = None, axes=None, show=False, prefix='') -> dict:
        return {}

    def sample(self, seed: chex.PRNGKey, sample_shape: chex.Shape) -> chex.Array:
        return None


if __name__ == "__main__":
    key = jax.random.PRNGKey(42)
    seeds = Seeds()

    samples = jax.random.normal(key, shape=(10, 26))
    print(samples)
    print(seeds.log_prob(samples))
    print(jax.vmap(seeds.log_prob)(samples))
