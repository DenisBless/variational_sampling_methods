"""
===========================================================================
The following code comes from the paper 'SE(3) Equivariant Augmented Coupling Flows'.
It is a modified version of the lennard jones potential to fit the structure of this library.
https://github.com/lollcat/se3-augmented-coupling-flows/tree/main

# BibTex
@inproceedings{
midgley2023eacf,
title={{SE}(3) Equivariant Augmented Coupling Flows},
author={Laurence Illing Midgley and Vincent Stimper and Javier Antoran and Emile Mathieu and Bernhard Sch{\"o}lkopf and Jos{\'e} Miguel Hern{\'a}ndez-Lobato},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023},
url={https://openreview.net/forum?id=KKxO6wwx8p}
}

===========================================================================
The code is licensed with the MIT License:
------------------------------------------

MIT License

Copyright (c) 2024 Laurence Midgley

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
===========================================================================
"""

from dataclasses import dataclass
from typing import Optional, Union, List

import chex
import jax.numpy as jnp
import jax
import jax.random
import matplotlib.pyplot as plt
from typing import Tuple

import wandb

from targets.base_target import Target


# ======================================================================================
# Copied from file 'se3-augmented-coupling-flows/eacf/utils/graph.py':

def get_senders_and_receivers_fully_connected(n_nodes: int) -> Tuple[chex.Array, chex.Array]:
    receivers = []
    senders = []
    for i in range(n_nodes):
        for j in range(n_nodes - 1):
            receivers.append(i)
            senders.append((i + 1 + j) % n_nodes)
    return jnp.array(senders), jnp.array(receivers)


# ======================================================================================
# Copied from file 'se3-augmented-coupling-flows/eacf/utils/numerical.py':

def safe_norm(x: jnp.ndarray, axis: int = None, keepdims=False) -> jnp.ndarray:
    """nan-safe norm. Copied from mace-jax"""
    x2 = jnp.sum(x ** 2, axis=axis, keepdims=keepdims)
    return jnp.where(x2 == 0, 1, x2) ** 0.5


# ======================================================================================
# Copied and !modified! from file 'se3-augmented-coupling-flows/eacf/targets/target_energy/leonard_jones.py':


@dataclass(frozen=True)
class LennardJonesParams:
    epsilon: float = 1.0
    tau: float = 1.0
    r: Union[float, chex.Array] = 1.0
    harmonic_potential_coef: float = 0.5


class LennardJones(Target):
    def __init__(self, dim: int, spatial_dim: int = 3, can_sample=False,
                 lj_params: Optional[LennardJonesParams] = None):
        """
        - dim: Number of nodes/atoms (Unfortunately the name 'dim' seems to be prescribed by
            how the config-system is set up and might not be changeable to a better name like 'n_nodes')
        - spatial_dim: Number of spatial dimensions e.g. 2 or 3

        The resulting total dimension of the target function will be target.dim = dim * spatial_dim
        """
        super().__init__(dim * spatial_dim, None, can_sample)

        self.spatial_dim = spatial_dim
        self.n_nodes = dim

        self.lj_params = lj_params if lj_params is not None else LennardJonesParams()

    def energy(self, x: chex.Array) -> chex.Array:
        epsilon = self.lj_params.epsilon
        tau = self.lj_params.tau
        r = self.lj_params.r
        harmonic_potential_coef = self.lj_params.harmonic_potential_coef

        chex.assert_rank(x, 2)
        if isinstance(r, float):
            r = jnp.ones(self.n_nodes) * r
        senders, receivers = get_senders_and_receivers_fully_connected(self.n_nodes)
        vectors = x[senders] - x[receivers]
        d = safe_norm(vectors, axis=-1)
        term_inside_sum = (r[receivers] / d) ** 12 - 2 * (r[receivers] / d) ** 6
        energy = epsilon / (2 * tau) * jnp.sum(term_inside_sum)

        # For harmonic potential see https://github.com/vgsatorras/en_flows/blob/main/deprecated/eqnode/test_systems.py#L94.
        # This oscillator is mentioned but not explicity specified in the paper where it was introduced:
        # http://proceedings.mlr.press/v119/kohler20a/kohler20a.pdf.
        centre_of_mass = jnp.mean(x, axis=0)
        harmonic_potential = harmonic_potential_coef * jnp.sum((x - centre_of_mass) ** 2)
        return energy + harmonic_potential

    def log_prob(self, x: chex.Array):
        if len(x.shape) == 1:
            return - self.energy(x.reshape((self.n_nodes, self.spatial_dim)))
        elif len(x.shape) == 2:
            batch = x.shape[0]
            return - jax.vmap(self.energy)(x.reshape((batch, self.n_nodes, self.spatial_dim)))
        else:
            raise Exception

    def visualise(self, samples: chex.Array, axes: List[plt.Axes] = None, show=False, prefix='') -> dict:
        plt.close()

        if len(samples.shape) == 1:
            samples = samples.reshape((1, self.n_nodes, self.spatial_dim))
        elif len(samples.shape) == 2:
            batch = samples.shape[0]
            samples = samples.reshape((batch, self.n_nodes, self.spatial_dim))
        else:
            raise Exception

        # Plot distance distribution of the first two nodes as a histogram
        dim = tuple(range(self.spatial_dim))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        d = jnp.linalg.norm(samples[:, 0, dim] - samples[:, 1, dim], axis=-1)
        ax1.hist(d, bins=50, density=True, alpha=0.4)

        # Scatter model samples in 2D or 3D
        project_to_2d = False  # Determine if 3D data is projected to 2D
        scatter_3d = (not project_to_2d) and (self.spatial_dim == 3)
        if scatter_3d:
            ax2 = fig.add_subplot(122, projection='3d')
            # Plot 3D projected node samples for each node
            for i in range(self.n_nodes):
                ax2.scatter(samples[:, i, 0].flatten(), samples[:, i, 1].flatten(), samples[:, i, 2].flatten(),
                            alpha=0.6, s=10)
        else:
            for i in range(self.n_nodes):
                ax2.scatter(samples[:, i, 0].flatten(), samples[:, i, 1].flatten(), alpha=0.6, s=10)

        # =========================================

        wb = {"figures/vis": [wandb.Image(fig)]}

        if show:
            plt.show()
        return wb

    def sample(self, seed, sample_shape):
        return None