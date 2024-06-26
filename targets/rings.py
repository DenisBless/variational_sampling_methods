import os

import jax
import jax.numpy as jnp
import chex
import wandb
from matplotlib import pyplot as plt
from targets.base_target import Target
from typing import List


class ConcentricRings(Target):
    def __init__(self, dim, log_Z=None, can_sample=True, sample_bounds=None) -> None:
        super().__init__(dim=dim, log_Z=log_Z, can_sample=can_sample)
        self.A = 16.1513
        self.c1 = 4.5 * jnp.pi
        self.c2 = (self.A - 1) / self.A

    def sample(self, seed: chex.PRNGKey, sample_shape: chex.Shape) -> chex.Array:
        dir = jax.random.normal(seed, (sample_shape[0], self.dim))
        dir = dir / jnp.linalg.norm(dir, axis=-1, keepdims=True)

        key, subkey = jax.random.split(seed)
        u = jax.random.uniform(subkey, minval=0, maxval=1, shape=sample_shape)

        rs = jnp.linspace(0, self.c1, 1000)
        r = jnp.where(u <= ((self.A - 1) / self.A),
                      rs[jnp.argmin(jnp.abs(jnp.sin(rs[:, None]) + 1.001 * rs[:, None] - self.A * u), 0)],
                      self.c1 - jnp.log(self.A * (1 - u))).reshape(-1, 1)

        return r * dir

    def log_prob(self, x: chex.Array) -> chex.Array:
        r = jnp.linalg.norm(x, axis=-1)
        lp = jnp.log(jnp.where(r <= self.c1, jnp.cos(r) + 1.001, jnp.exp((self.c1 - r))))
        return jnp.where(lp == -jnp.inf, -100, lp)

    def visualise(self, samples: chex.Array = None, axes=None, show=False, prefix='') -> dict:
        if self.dim == 2:
            fig = plt.figure()
            ax = fig.add_subplot()
            x, y = jnp.meshgrid(jnp.linspace(-20, 20, 300), jnp.linspace(-20, 20, 300))
            grid = jnp.c_[x.ravel(), y.ravel()]
            pdf_values = jax.vmap(jnp.exp)(self.log_prob(grid))
            pdf_values = jnp.reshape(pdf_values, x.shape)
            ax.contourf(x, y, pdf_values, levels=20, cmap='viridis')
            if samples is not None:
                plt.scatter(samples[:, 0], samples[:, 1], c='r', alpha=0.5, marker='x')
            # plt.xlabel('X')
            # plt.ylabel('Y')
            # plt.colorbar()
            plt.xticks([])
            plt.yticks([])
            # plt.savefig(os.path.join(project_path('./figures/'), f"rings2D.pdf"), bbox_inches='tight', pad_inches=0.1)
            wb = {"figures/vis": [wandb.Image(fig)]}
            if show:
                plt.show()

            return wb

        else:
            return {}


if __name__ == "__main__":
    key = jax.random.PRNGKey(45)
    cr = ConcentricRings(dim=2)
    print(cr.sample(key, (2000,)))
    cr.visualise(samples=None, show=True)
