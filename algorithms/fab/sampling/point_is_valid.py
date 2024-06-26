"""Code builds on https://github.com/lollcat/fab-jax"""
from typing import Protocol, Union

import chex
import jax.numpy as jnp

from algorithms.fab.sampling.base import Point

class PointIsValidFn(Protocol):
    def __call__(self, point: Point) -> bool:
        """
        Determines whether a point is valid or invalid. A point may be invalid if it contains
        NaN values under the target log prob. The user can provide any criternion for this function, which allows
        for enforcement of additional useful properties, such as problem bounds.
        See `default_point_is_valid` for the default version of this function.
        """

def default_point_is_valid_fn(point: Point) -> bool:
    chex.assert_rank(point.x, 1)
    is_valid = jnp.isfinite(point.log_q) & jnp.isfinite(point.log_p) & jnp.all(jnp.isfinite(point.x))
    return is_valid


def point_is_valid_if_in_bounds_fn(point: Point,
                                   min_bounds: Union[chex.Array, float],
                                   max_bounds: [chex.Array, float]) -> bool:
    """Returns True if a point is within the provided bounds. Must be wrapped with a partial to be
    used as a `PointIsValidFn`."""
    chex.assert_rank(point.x, 1)
    if isinstance(min_bounds, chex.Array):
        chex.assert_equal_shape(point.x, min_bounds, max_bounds)
    else:
        min_bounds = jnp.ones_like(point.x) * min_bounds
        max_bounds = jnp.ones_like(point.x) * max_bounds

    is_valid = (point.x > min_bounds).all() & (point.x < max_bounds).all()
    is_valid = is_valid & default_point_is_valid_fn(point)
    return is_valid
