from typing import List

import chex
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
import pickle
import numpyro
import numpyro.distributions as pydist
from jax._src.flatten_util import ravel_pytree

from targets.base_target import Target
from utils.path_utils import project_path


def pad_with_const(X):
    extra = np.ones((X.shape[0], 1))
    return np.hstack([extra, X])


def standardize_and_pad(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1.
    X = (X - mean) / std
    return pad_with_const(X)


def load_model_ionosphere():
    def model(Y):
        w = numpyro.sample("weights", pydist.Normal(jnp.zeros(dim), jnp.ones(dim)))
        logits = jnp.dot(X, w)
        with numpyro.plate('J', n_data):
            y = numpyro.sample("y", pydist.BernoulliLogits(logits), obs=Y)

    with open(project_path('targets/data/ionosphere_full.pkl'), 'rb') as f:
        X, Y = pickle.load(f)

    Y = (Y + 1) // 2
    X = standardize_and_pad(X)

    dim = X.shape[1]
    n_data = X.shape[0]
    model_args = (Y,)
    return model, model_args


class Ionosphere(Target):
    def __init__(self, dim=35, log_Z=None, can_sample=False, sample_bounds=None) -> None:
        super().__init__(dim=dim, log_Z=log_Z, can_sample=can_sample)
        self.data_ndim = dim

        rng_key = jax.random.PRNGKey(1)
        model, model_args = load_model_ionosphere()
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
