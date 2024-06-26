import distrax
import jax
import jax.numpy as jnp
import chex
import jax.random as random
import matplotlib
import numpy as np
import numpyro.distributions as dist
import wandb
from scipy.stats import wishart
from matplotlib import pyplot as plt
from targets.base_target import Target
from typing import List

from utils.path_utils import project_path


# matplotlib.use('agg')


class GaussianMixtureModel(Target):
    def __init__(self, num_components, dim, log_Z=0., can_sample=True, sample_bounds=None) -> None:
        # parameters
        super().__init__(dim, log_Z, can_sample)

        self.num_components = num_components

        # parameters
        min_mean_val = -10
        max_mean_val = 10
        min_val_mixture_weight = 0.3
        max_val_mixture_weight = 0.7
        degree_of_freedom_wishart = dim + 2

        seed = jax.random.PRNGKey(0)

        # set mixture components
        locs = jax.random.uniform(seed, minval=min_mean_val, maxval=max_mean_val, shape=(num_components, dim))
        covariances = []
        for _ in range(num_components):
            seed, subkey = random.split(seed)

            # Set the random seed for Scipy
            seed_value = random.randint(key=subkey, shape=(), minval=0, maxval=2 ** 30)
            np.random.seed(seed_value)

            cov_matrix = wishart.rvs(df=degree_of_freedom_wishart, scale=jnp.eye(dim))
            covariances.append(cov_matrix)

        component_dist = distrax.MultivariateNormalFullCovariance(locs, jnp.array(covariances))

        # set mixture weights
        uniform_mws = True
        if uniform_mws:
            mixture_weights = distrax.Categorical(logits=jnp.ones(num_components) / num_components)
        else:
            mixture_weights = distrax.Categorical(logits=dist.Uniform(
                low=min_val_mixture_weight, high=max_val_mixture_weight).sample(seed, sample_shape=(num_components,)))

        self.mixture_distribution = distrax.MixtureSameFamily(mixture_distribution=mixture_weights,
                                                              components_distribution=component_dist)

    def sample(self, seed: chex.PRNGKey, sample_shape: chex.Shape) -> chex.Array:
        return self.mixture_distribution.sample(seed=seed, sample_shape=sample_shape)

    def log_prob(self, x: chex.Array) -> chex.Array:
        batched = x.ndim == 2

        if not batched:
            x = x[None,]

        log_prob = self.mixture_distribution.log_prob(x)

        if not batched:
            log_prob = jnp.squeeze(log_prob, axis=0)

        return log_prob

    def entropy(self, samples: chex.Array = None):
        expanded = jnp.expand_dims(samples, axis=-2)
        # Compute `log_prob` in every component.
        idx = jnp.argmax(self.mixture_distribution.components_distribution.log_prob(expanded), 1)
        unique_elements, counts = jnp.unique(idx, return_counts=True)
        mode_dist = counts / samples.shape[0]
        entropy = -jnp.sum(mode_dist * (jnp.log(mode_dist) / jnp.log(self.num_components)))
        return entropy

    def visualise(self, samples: chex.Array = None, axes=None, show=False, prefix='') -> dict:
        plt.close()

        boarder = [-15, 15]

        fig = plt.figure()
        ax = fig.add_subplot()

        if self.dim == 2:

            x, y = jnp.meshgrid(jnp.linspace(boarder[0], boarder[1], 100),
                                jnp.linspace(boarder[0], boarder[1], 100))
            grid = jnp.c_[x.ravel(), y.ravel()]
            pdf_values = jax.vmap(jnp.exp)(self.log_prob(grid))
            pdf_values = jnp.reshape(pdf_values, x.shape)
            ax.contourf(x, y, pdf_values, levels=20)  # , cmap='viridis')
            if samples is not None:
                plt.scatter(samples[:300, 0], samples[:300, 1], c='r', alpha=0.5, marker='x')
            # plt.xlabel('X')
            # plt.ylabel('Y')
            # plt.colorbar()
            plt.xticks([])
            plt.yticks([])
            # plt.savefig(os.path.join(project_path('./figures/'), f"gmm2D.pdf"), bbox_inches='tight', pad_inches=0.1)

            try:
                wandb.log({"images/target_vis": wandb.Image(plt)})
            except:
                pass

            # import tikzplotlib
            # import os
            # plt.savefig(os.path.join(project_path('./figures/'), f"gmm.pdf"), bbox_inches='tight', pad_inches=0.1)
            # tikzplotlib.save(os.path.join(project_path('./figures/'), f"gmm.tex"))

        else:
            target_samples = self.sample(jax.random.PRNGKey(0), (500,))
            ax.scatter(target_samples[:, 0], target_samples[:, 1], c='b', label='target')
            ax.scatter(samples[:, 0], samples[:, 1], c='r', label='model')
            plt.legend()

        wb = {"figures/vis": [wandb.Image(fig)]}
        if show:
            plt.show()

        return wb


if __name__ == "__main__":
    key = jax.random.PRNGKey(45)
    gmm = GaussianMixtureModel(dim=2)
    samples = gmm.sample(key, (1000,))
    print(gmm.entropy(samples))
    # sample = gmm.sample(key, (1,))
    # print(sample)
    # print(samples)
    # print((gmm.log_prob(sample)).shape)
    # print((jax.vmap(gmm.log_prob)(sample)).shape)
    gmm.visualise(show=True)
