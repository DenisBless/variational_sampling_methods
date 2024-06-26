import os

import jax
import jax.numpy as jnp
import chex
import jax.random as random
import pandas as pd
import numpy as np
import numpyro.distributions as dist
import wandb
from scipy.stats import wishart
from matplotlib import pyplot as plt
from targets.base_target import Target
from typing import List

from utils.path_utils import project_path


class Gaussian(Target):
    def __init__(self, dim, log_Z=0., can_sample=True, sample_bounds=None) -> None:
        super().__init__(dim, log_Z, can_sample)

        # parameters
        min_mean_val = -1
        max_mean_val = 1
        degree_of_freedom_wishart = dim + 2

        seed = jax.random.PRNGKey(0)

        # set mixture components
        locs = jax.random.uniform(seed, minval=min_mean_val, maxval=max_mean_val, shape=(dim,))
        seed, subkey = random.split(seed)

        # Set the random seed for Scipy
        seed_value = random.randint(key=subkey, shape=(), minval=0, maxval=2 ** 30)
        np.random.seed(seed_value)

        cov_matrix = wishart.rvs(df=degree_of_freedom_wishart, scale=jnp.eye(dim))

        self.pdf = dist.MultivariateNormal(locs, jnp.array(cov_matrix))
        # self.pdf = dist.MultivariateNormal(jnp.zeros(self.dim), 2.5 **2 * jnp.eye(self.dim))

    def sample(self, seed: chex.PRNGKey, sample_shape: chex.Shape) -> chex.Array:
        return self.pdf.sample(key=seed, sample_shape=sample_shape)

    def log_prob(self, x: chex.Array) -> chex.Array:
        log_prob = self.pdf.log_prob(x)
        return log_prob

    def visualise(self, samples: chex.Array = None, axes: List[plt.Axes] = None, show=False, clip=False) -> None:
        plt.close()
        boarder = [-15, 15]

        if self.dim == 2:
            fig = plt.figure()
            ax = fig.add_subplot()

            x, y = jnp.meshgrid(jnp.linspace(boarder[0], boarder[1], 100),
                                jnp.linspace(boarder[0], boarder[1], 100))
            grid = jnp.c_[x.ravel(), y.ravel()]
            pdf_values = jax.vmap(jnp.exp)(self.log_prob(grid))
            pdf_values = jnp.reshape(pdf_values, x.shape)
            ax.contourf(x, y, pdf_values, levels=20, cmap='viridis')
            if samples is not None:
                plt.scatter(samples[:300, 0], samples[:300, 1], c='r', alpha=0.5, marker='x')
            # plt.xlabel('X')
            # plt.ylabel('Y')
            # plt.colorbar()
            # plt.xticks([])
            # plt.yticks([])
            # plt.savefig(os.path.join(project_path('./figures/'), f"gmm2D.pdf"), bbox_inches='tight', pad_inches=0.1)
            wb = {"figures/vis": [wandb.Image(fig)]}
            if show:
                plt.show()

            return wb

        else:

            return {}


if __name__ == "__main__":
    key = jax.random.PRNGKey(45)
    gmm = Gaussian(dim=2)
    # print(gmm.sample(key, (2,)))
    gmm.visualise(show=True)
