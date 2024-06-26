"""Code builds on https://github.com/lollcat/fab-jax"""
from typing import Sequence, Callable

import jax
import flax.linen as nn

class ConditionerMLP(nn.Module):
    """Used for converting the invariant feat from the EGNN, into the parameters of the bijector transformation
    (e.g. scale and shit params for RealNVP)."""
    name: str
    mlp_units: Sequence[int]
    n_output_params: int
    zero_init: bool
    activation: Callable = jax.nn.gelu

    @nn.compact
    def __call__(self, params):
        out = params
        for unit in self.mlp_units:
            out = nn.Dense(unit, param_dtype=params.dtype)(out)
            out = self.activation(out)

        out = nn.Dense(self.n_output_params,
                       kernel_init=nn.initializers.zeros_init() if self.zero_init else
                       nn.initializers.variance_scaling(scale=0.01, mode="fan_in", distribution="truncated_normal"),
                       param_dtype=params.dtype
                       )(out)
        return out
