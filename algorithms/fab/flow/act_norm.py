"""Code builds on https://github.com/lollcat/fab-jax"""
from typing import Tuple, Callable

import distrax
import flax.linen as nn
import jax.numpy as jnp
import jax
import chex
import tensorflow_probability.substrates.jax as tfp

from algorithms.fab.utils.jax_util import inverse_softplus

class UnconditionalAffine(distrax.Bijector):
    """Unconditioned Affine transform."""
    def __init__(self, get_params: Callable[[], Tuple[chex.Array, chex.Array]],
                 restrict_scaling: bool = True) -> None:
        super().__init__(event_ndims_in = 1,
               is_constant_jacobian = True,
               is_constant_log_det = True)

        self._get_params = get_params
        self._restrict_scaling = restrict_scaling

    def get_params(self):
        scale_logit, shift = self._get_params()
        if self._restrict_scaling:
            # Prevents any individual act norm layer doing a huge amount of scaling.
            scale_logit_bijector = tfp.bijectors.Sigmoid(low=0.02, high=50.)
            scale_logit_init = scale_logit_bijector.inverse(1.)
            scale = scale_logit_bijector(scale_logit + scale_logit_init)
        else:
            scale = jax.nn.softplus(scale_logit + inverse_softplus(1.))
        shift = shift
        return scale, shift

    def forward_and_log_det(self, x: chex.Array) -> Tuple[chex.Array, chex.Array]:
        scale, shift = self.get_params()

        if x.ndim == 2:
            scale = jnp.ones_like(x) * scale[None, :]
            shift = jnp.ones_like(x) * shift[None, :]
        else:
            assert x.ndim == 1
        bijector = distrax.Block(distrax.ScalarAffine(shift=shift, scale=scale), ndims=1)
        return bijector.forward_and_log_det(x)

    def inverse_and_log_det(self, y: chex.Array) -> Tuple[chex.Array, chex.Array]:
        scale, shift = self.get_params()

        if y.ndim == 2:
            scale = jnp.ones_like(y) * scale[None, :]
            shift = jnp.ones_like(y) * shift[None, :]
        else:
            assert y.ndim == 1
        bijector = distrax.Block(distrax.ScalarAffine(shift=shift, scale=scale), ndims=1)
        return bijector.inverse_and_log_det(y)


def build_act_norm_layer(
    dim: int,
    identity_init: bool,
    restrict_scaling: bool = True,  # Similar to real_nvp we can restrict the scaling size of act norm to improve stability.
):
    assert identity_init is True, "Only this option currently configured."

    class GetScaleShift(nn.Module):
        @nn.compact
        def __call__(self) -> Tuple[chex.Array, chex.Array]:
            scale_logit = self.param("scale_logit", nn.initializers.zeros_init(), (dim,))
            shift = self.param("shift", nn.initializers.zeros_init(), (dim,))
            return scale_logit, shift

    def get_scale_and_shift():
        scale_shift_module = GetScaleShift()
        scale_logit, shift = scale_shift_module()
        return scale_logit, shift

    return UnconditionalAffine(get_scale_and_shift, restrict_scaling=restrict_scaling)
