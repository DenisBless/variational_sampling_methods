import os

import distrax
import jax
import jax.numpy as jnp
import chex
import pandas as pd
import numpyro.distributions as dist
from matplotlib import pyplot as plt
from targets.base_target import Target
from typing import List
import wandb

from utils.path_utils import project_path


class StudentTMixtureModel(Target):
    def __init__(self, num_components, dim, log_Z=0., can_sample=True, sample_bounds=None) -> None:
        # parameters
        super().__init__(dim, log_Z, can_sample)
        seed = 0
        self.num_components = num_components

        # parameters
        min_mean_val = -10
        max_mean_val = 10
        min_val_mixture_weight = 0.3
        max_val_mixture_weight = 0.7
        degree_of_freedoms = 2

        seed = jax.random.PRNGKey(seed)

        # set mixture components
        locs = jax.random.uniform(seed, minval=min_mean_val, maxval=max_mean_val, shape=(num_components, dim))
        dofs = jnp.ones((num_components, dim)) * degree_of_freedoms
        scales = jnp.ones((num_components, dim))
        component_dist = dist.Independent(dist.StudentT(df=dofs, loc=locs, scale=scales), 1)

        # component_dist = dist.MultivariateStudentT(df=dofs, loc=locs,
        #                                           scale_tril=jnp.array([jnp.tril(jnp.diag(row)) for row in scales]),
        #                                           validate_args=True)

        uniform_mws = True
        if uniform_mws:
            mixture_weights = dist.Categorical(logits=jnp.ones(num_components) / num_components)
        else:
            mixture_weights = dist.Categorical(logits=dist.Uniform(
                low=min_val_mixture_weight, high=max_val_mixture_weight).sample(seed, sample_shape=(num_components,)))

        self.mixture_distribution = dist.MixtureSameFamily(mixture_weights, component_dist)

    def sample(self, seed: chex.PRNGKey, sample_shape: chex.Shape) -> chex.Array:
        return self.mixture_distribution.sample(key=seed, sample_shape=sample_shape)

    def log_prob(self, x: chex.Array) -> chex.Array:
        batched = x.ndim == 2

        if not batched:
            x = x[None,]

        log_prob = self.mixture_distribution.log_prob(x)

        if not batched:
            log_prob = jnp.squeeze(log_prob, axis=0)

        return log_prob

    def entropy(self, samples: chex.Array = None):
        idx = jnp.argmax(self.mixture_distribution.component_log_probs(samples), 1)
        unique_elements, counts = jnp.unique(idx, return_counts=True)
        mode_dist = counts / samples.shape[0]
        entropy = -jnp.sum(mode_dist * (jnp.log(mode_dist) / jnp.log(self.num_components)))
        return entropy

    def visualise(self, samples: chex.Array = None, axes=None, show=False, prefix='') -> dict:
        boarder = [-15, 15]
        if self.dim == 2:
            fig = plt.figure()
            ax = fig.add_subplot()
            x, y = jnp.meshgrid(jnp.linspace(boarder[0], boarder[1], 100),
                                jnp.linspace(boarder[0], boarder[1], 100))
            grid = jnp.c_[x.ravel(), y.ravel()]
            pdf_values = jax.vmap(jnp.exp)(self.log_prob(grid))
            pdf_values = jnp.reshape(pdf_values, x.shape)
            # ax.contourf(x, y, pdf_values, levels=20, cmap='viridis')
            ax.contourf(x, y, pdf_values, levels=50)

            if samples is not None:
                plt.scatter(samples[:300, 0], samples[:300, 1], c='r', alpha=0.5, marker='x')

            # plt.xlabel('X')
            # plt.ylabel('Y')
            plt.xticks([])
            plt.yticks([])
            wb = {"figures/vis": [wandb.Image(fig)]}
            if show:
                plt.show()

            return wb

        else:
            return {}

            # import tikzplotlib
            # plt.savefig(os.path.join(project_path('./figures/'), f"stmm.pdf"), bbox_inches='tight', pad_inches=0.1)
            # tikzplotlib.save(os.path.join(project_path('./figures/'), f"stmm.tex"))


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    stmm = StudentTMixtureModel(dim=2)
    sample = stmm.sample(key, (5,))
    print(stmm.entropy(sample))
    row_indices, column_indices = jnp.where(jnp.abs(sample) > 100)
    # print(sample[row_indices, column_indices])
    # print(stmm.log_prob(sample))
    stmm.visualise(show=True)
