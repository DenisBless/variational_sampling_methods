import os

import wandb
from typing import List
import jax.numpy as jnp
import distrax
import chex
import jax.random
import matplotlib.pyplot as plt
import numpy as np
from targets.base_target import Target


class GMM1D(Target):
    def __init__(self, dim=1, log_Z=0., can_sample=True, sample_bounds=None) -> None:
        super().__init__(dim, log_Z, can_sample)


        self.num_comp = 4
        logits = jnp.ones(self.num_comp)
        mean = jnp.array([-3.0, 0.5, 1.5, 2.5]).reshape((-1, 1))
        scale = jnp.array([0.5, 0.6, 0.3, 0.4]).reshape((-1, 1))

        mixture_dist = distrax.Categorical(logits=logits)
        components_dist = distrax.Independent(
            distrax.Normal(loc=mean, scale=scale),
        )
        self.distribution = distrax.MixtureSameFamily(
            mixture_distribution=mixture_dist,
            components_distribution=components_dist,
        )

    def log_prob(self, x: chex.Array) -> chex.Array:
        if x.ndim == 0:
            x = jnp.expand_dims(x, axis=0)

        assert x.shape[-1] == 1, f"The last dimension of x should be 1, but got {x.shape[-1]}"

        batched = x.ndim == 2

        if not batched:
            x = jnp.expand_dims(x, axis=1)

        log_prob = self.distribution.log_prob(x)

        if not batched:
            log_prob = jnp.squeeze(log_prob, axis=0)

        return log_prob

    def sample(self, seed: chex.PRNGKey, sample_shape: chex.Shape = ()) -> chex.Array:
        return self.distribution.sample(seed=seed, sample_shape=sample_shape)

    def entropy(self, samples: chex.Array = None):
        expanded = jnp.expand_dims(samples, axis=-2)
        # Compute `log_prob` in every component.
        idx = jnp.argmax(self.distribution.components_distribution.log_prob(expanded), 1)
        unique_elements, counts = jnp.unique(idx, return_counts=True)
        mode_dist = counts / samples.shape[0]
        entropy = -jnp.sum(mode_dist * (jnp.log(mode_dist) / jnp.log(self.num_comp)))
        return entropy

    def visualise(self, samples: List[chex.Array] = None, axes: List[plt.Axes] = None,
                  show=False, suffix: str = ''):
        x_range = (-5, 5)
        resolution = 100

        y_grid = np.linspace(x_range[0], x_range[1], resolution + 1)
        marg_dens, _ = np.histogram(samples, bins=y_grid, density=True)
        dark_gray = np.array([[90 / 255, 90 / 255, 90 / 255, 1.0]])

        y_range = (y_grid[0], y_grid[-1])
        # plot
        fig, ax = plt.subplots(1, 1, figsize=(6, 4),)

        ax.set_xlim(*y_range)
        ax.set_xlabel('$x$')
        # ax.set_title('$\\hat{p}(x_T)$')
        ax.hist(y_grid[:-1], weights=marg_dens, range=y_range, bins=y_grid, color=dark_gray[0],
                   orientation='vertical', edgecolor='white', linewidth=0.75)
        # Generate x values for the function plot


        x_values = np.linspace(*x_range, 1000)
        x_values = np.expand_dims(x_values, 1)
        # Compute y values using the function g(x)
        log_probs = jnp.exp(self.log_prob(x_values))

        # Plot the function g(x) on the same axes
        ax.plot(x_values, log_probs, label='$g(x)$', color='black')

        wb = {"figures/vis": wandb.Image(fig)}
        if show:
            plt.show()

        path = './utils/images/'
        name = suffix + '_diff_traj'
        plt.savefig(os.path.join(path, name + '.png'), bbox_inches='tight', pad_inches=0.1, dpi=300)

        plt.close()

        return wb


if __name__ == '__main__':
    gmm = GMM1D()
    # one component, 40 bathc, 60 time
    samples = gmm.sample(jax.random.PRNGKey(0), (10000,))
    gmm.log_prob(samples)
    jax.vmap(gmm.log_prob)(samples)
    gmm.entropy(samples)
    # gmm.visualise( show=True)
    gmm.visualise(samples, show=True)