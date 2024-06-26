from typing import List

import chex
import jax
import jax.numpy as jnp
import distrax
from matplotlib import pyplot as plt
import wandb

from targets.base_target import Target
from algorithms.fab.utils.plot import plot_marginal_pair, plot_contours_2D
from utils.path_utils import project_path
import matplotlib


# matplotlib.use('agg')

class GMM40(Target):
    def __init__(
            self,
            dim: int = 2, num_components: int = 40, loc_scaling: float = 40,
            scale_scaling: float = 1.0, seed: int = 0, sample_bounds=None, can_sample=True, log_Z=0
    ) -> None:
        super().__init__(dim, log_Z, can_sample)

        self.seed = seed
        self.n_mixes = num_components

        key = jax.random.PRNGKey(seed)
        logits = jnp.ones(num_components)
        mean = jax.random.uniform(shape=(num_components, dim), key=key, minval=-1.0, maxval=1.0) * loc_scaling
        scale = jnp.ones(shape=(num_components, dim)) * scale_scaling

        mixture_dist = distrax.Categorical(logits=logits)
        components_dist = distrax.Independent(
            distrax.Normal(loc=mean, scale=scale), reinterpreted_batch_ndims=1
        )
        self.distribution = distrax.MixtureSameFamily(
            mixture_distribution=mixture_dist,
            components_distribution=components_dist,
        )

        self._plot_bound = loc_scaling * 1.5

    def log_prob(self, x: chex.Array) -> chex.Array:
        batched = x.ndim == 2
        if not batched:
            x = x[None,]

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
        entropy = -jnp.sum(mode_dist * (jnp.log(mode_dist) / jnp.log(self.n_mixes)))
        return entropy

    def visualise(self, samples: chex.Array = None, axes=None, show=False, prefix='') -> dict:
        plt.close()
        fig = plt.figure()
        ax = fig.add_subplot()
        if samples is not None:
            plot_marginal_pair(samples[:, :2], ax, bounds=(-self._plot_bound, self._plot_bound))
            # jnp.save(project_path(f'samples/gmm40_samples'), samples)
        if self.dim == 2:
            plot_contours_2D(self.log_prob, ax, bound=self._plot_bound, levels=50)
        plt.xticks([])
        plt.yticks([])
        # import os
        # plt.savefig(os.path.join(project_path('./samples/gaussian_mixture40'), f"{prefix}gmm40.pdf"), bbox_inches='tight', pad_inches=0.1)

        wb = {"figures/vis": [wandb.Image(fig)]}
        if show:
            plt.show()

        return wb
        # import tikzplotlib
        # import os
        # tikzplotlib.save(os.path.join(project_path('./figures/'), f"gmm40.tex"))


if __name__ == '__main__':
    gmm = GMM40()
    samples = gmm.sample(jax.random.PRNGKey(0), (2000,))
    gmm.log_prob(samples)
    gmm.entropy(samples)
    # gmm.visualise( show=True)
    gmm.visualise(show=True)
