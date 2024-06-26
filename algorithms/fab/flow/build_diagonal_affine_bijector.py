"""Code builds on https://github.com/lollcat/fab-jax"""
import chex
import distrax
import flax.linen as nn
import jax.numpy as jnp

from algorithms.fab.flow.distrax_with_extra import SplitCouplingWithExtra, ChainWithExtra


class ConstantParametersFn(nn.Module):
    name: str
    n_output_params: int

    @nn.compact
    def __call__(self):
        params = self.param(
            'parameters', nn.initializers.zeros_init(),
            (self.n_output_params,), jnp.float32
        )
        return params


def build_diagonal_affine_bijector(
        dim: int,
        transform_type: str = 'diagonal_affine',
):
    assert transform_type in ['diagonal_affine']

    split_index = dim

    def make_conditioner(name, n_output_params):
        def conditioner(x: chex.Array) -> chex.Array:
            params = ConstantParametersFn(name=name, n_output_params=n_output_params)()
            if x.ndim != 1:
                assert x.ndim == 2
                params = jnp.repeat(params[None], x.shape[0], axis=0)
            return params

        return conditioner

    conditioner = make_conditioner(f'conditioner_for_linear_affine', split_index)

    def bijector_fn(params: chex.Array) -> distrax.Bijector:
        log_scale, shift = jnp.split(params, 2, axis=-1)
        return distrax.ScalarAffine(shift=shift, scale=log_scale)


    bijector = SplitCouplingWithExtra(
        split_index=split_index,
        event_ndims=1,
        conditioner=conditioner,
        bijector=bijector_fn,
    )

    return ChainWithExtra([bijector])