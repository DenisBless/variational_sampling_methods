"""Code builds on https://github.com/lollcat/fab-jax"""
from typing import Sequence

import chex
import distrax
import jax.nn
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp

from algorithms.fab.flow.distrax_with_extra import SplitCouplingWithExtra, ChainWithExtra, BijectorWithExtra
from algorithms.fab.utils.nets import ConditionerMLP
from algorithms.fab.utils.jax_util import inverse_softplus


def make_conditioner(name, n_output_params, mlp_units, identity_init):
    def conditioner(x: chex.Array) -> chex.Array:
        mlp = ConditionerMLP(name=name, mlp_units=mlp_units,
                             n_output_params=n_output_params,
                             zero_init=identity_init)
        if x.ndim == 1:
            params = mlp(x[None, :])
            params = jnp.squeeze(params, axis=0)
        else:
            params = mlp(x)
        return params

    return conditioner


def build_split_coupling_bijector(
        dim: int,
        identity_init: bool,
        conditioner_mlp_units: Sequence[int],
        transform_type: str = 'spline',
        restrict_scale_rnvp: bool = True,  # Hugely improves stability, strong reccomend.
        spline_max: float = 10.,
        spline_min: float = -10.,
        spline_num_bins: int = 8,
) -> BijectorWithExtra:
    assert transform_type in ['real_nvp', 'spline']

    split_index = dim // 2

    bijectors = []
    for swap in (True, False):
        params_after_split = dim - split_index
        params_transformed = split_index if swap else params_after_split
        if transform_type == "real_nvp":
            conditioner_n_params_out = params_transformed * 2
        elif transform_type == "spline":
            conditioner_n_params_out = params_transformed * (3 * spline_num_bins + 1)
        else:
            raise NotImplementedError

        conditioner = make_conditioner(f'splitcoupling_conditioner_swap{swap}',
                                       conditioner_n_params_out,
                                       conditioner_mlp_units, identity_init)

        def bijector_fn(params: chex.Array) -> distrax.Bijector:
            if transform_type == "real_nvp":
                scale_logit, shift = jnp.split(params, 2, axis=-1)
                if restrict_scale_rnvp:
                    scale_logit_bijector = tfp.bijectors.Sigmoid(low=0.1, high=10.)
                    scale_logit_init = scale_logit_bijector.inverse(1.)
                    scale = scale_logit_bijector(scale_logit + scale_logit_init)
                else:
                    scale = jax.nn.softplus(scale_logit + inverse_softplus(1.))
                return distrax.ScalarAffine(shift=shift, scale=scale)
            elif transform_type == "spline":
                params = jnp.reshape(params, (*params.shape[:-1], params_transformed, 3 * spline_num_bins + 1))
                bijector = distrax.RationalQuadraticSpline(
                    params=params,
                    range_min=spline_min,
                    range_max=spline_max,
                    min_bin_size=1e-4 * (spline_max - spline_min),
                    boundary_slopes='unconstrained'
                )
                return bijector
            else:
                raise NotImplementedError

        bijector = SplitCouplingWithExtra(
            split_index=split_index,
            event_ndims=1,
            conditioner=conditioner,
            bijector=bijector_fn,
            swap=swap,
            split_axis=-1
        )
        bijectors.append(bijector)

    return ChainWithExtra(bijectors)
