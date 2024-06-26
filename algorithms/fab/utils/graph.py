"""Code builds on https://github.com/lollcat/fab-jax"""
from typing import Callable, Tuple

import chex
import jax.numpy as jnp
import numpy as np

def setup_flat_log_prob(log_prob_fn: Callable, event_shape: chex.Shape) -> Tuple[Callable, Callable, Callable]:
    """Often it's easier if we only have one event dimension.
    This util sets up common functions when dealing with flattening a log prob function that takes in
     multiple event dims."""
    def flatten(x: chex.Array) -> chex.Array:
        return jnp.reshape(x, (*x.shape[:-len(event_shape)], np.prod(event_shape)))

    def unflatten(x: chex.Array) -> chex.Array:
        return jnp.reshape(x, (*x.shape[:-1], *event_shape))

    def flat_log_prob_fn(x: chex.Array) -> chex.Array:
        """Takes in flat x."""
        return log_prob_fn(unflatten(x))

    return flatten, unflatten, flat_log_prob_fn