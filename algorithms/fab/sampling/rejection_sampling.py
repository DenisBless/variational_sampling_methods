"""Code builds on https://github.com/lollcat/fab-jax"""
from typing import Callable

import chex
import distrax
import jax.random
import jax.numpy as jnp


def rejection_sampling(n_samples: int,
                       proposal: distrax.Distribution,
                       target_log_prob_fn: Callable,
                       k: float,
                       key: chex.PRNGKey) -> chex.Array:
    """Rejection sampling. See Pattern Recognition and ML by Bishop Chapter 11.1"""
    # Note: This currently is not written to work inside of jax.jit or jax.vmap.
    key1, key2, key3 = jax.random.split(key, 3)
    n_samples_propose = n_samples*10
    z_0, log_q_z0 = proposal._sample_n_and_log_prob(key, n=n_samples_propose)
    u_0 = jax.random.uniform(key=key2, shape=(n_samples_propose,)) * k*jnp.exp(log_q_z0)
    accept = jnp.exp(target_log_prob_fn(z_0)) > u_0
    samples = z_0[accept]
    if samples.shape[0] >= n_samples:
        return samples[:n_samples]
    else:
        required_samples = n_samples - samples.shape[0]
        new_samples = rejection_sampling(required_samples, proposal, target_log_prob_fn, k, key3)
        samples = jnp.concatenate([samples, new_samples], axis=0)
        return samples
