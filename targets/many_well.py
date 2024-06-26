from typing import List

import jax.numpy as jnp
import jax
import chex
# import matplotlib
#
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import wandb

from targets.base_target import Target


# Taken from FAB code
class Energy:
    """
    https://zenodo.org/record/3242635#.YNna8uhKjIW
    """

    def __init__(self, dim):
        super().__init__()
        self._dim = dim

    @property
    def dim(self):
        return self._dim

    def _energy(self, x):
        raise NotImplementedError()

    def energy(self, x, temperature=None):
        assert x.shape[-1] == self._dim, "`x` does not match `dim`"
        if temperature is None:
            temperature = 1.
        return self._energy(x) / temperature

    def force(self, x, temperature=None):
        e_func = lambda x: jnp.sum(self.energy(x, temperature=temperature))
        return -jax.grad(e_func)(x)


class DoubleWellEnergy(Energy):
    def __init__(self, a: float = -0.5, b: float = -6.0, c: float = 1.):
        dim = 2
        super().__init__(dim)
        self._a = a
        self._b = b
        self._c = c

    def _energy(self, x):
        d = x[:, [0]]
        v = x[:, 1:]
        e1 = self._a * d + self._b * d ** 2 + self._c * d ** 4
        e2 = jnp.sum(0.5 * v ** 2, axis=-1, keepdims=True)
        return e1 + e2

    def log_prob(self, x):
        if len(x.shape) == 1:
            x = jnp.expand_dims(x, axis=0)
        return jnp.squeeze(-self.energy(x))

    @property
    def log_Z(self):
        log_Z_dim0 = jnp.log(11784.50927)
        log_Z_dim1 = 0.5 * jnp.log(2 * jnp.pi)
        return log_Z_dim0 + log_Z_dim1


class ManyWellEnergy(Target):
    def __init__(self, a: float = -0.5, b: float = -6.0, c: float = 1., dim=32, can_sample=False, sample_bounds=None) -> None:
        assert dim % 2 == 0
        self.n_wells = dim // 2
        self.double_well_energy = DoubleWellEnergy(a, b, c)

        log_Z = self.double_well_energy.log_Z * self.n_wells
        super().__init__(dim=dim, log_Z=log_Z, can_sample=can_sample)

        self.centre = 1.7
        self.max_dim_for_all_modes = 40  # otherwise we get memory issues on huge test set
        if self.dim < self.max_dim_for_all_modes:
            dim_1_vals_grid = jnp.meshgrid(*[jnp.array([-self.centre, self.centre]) for _ in
                                             range(self.n_wells)])
            dim_1_vals = jnp.stack([dim.flatten() for dim in dim_1_vals_grid], axis=-1)
            n_modes = 2 ** self.n_wells
            assert n_modes == dim_1_vals.shape[0]
            test_set = jnp.zeros((n_modes, dim))
            test_set = test_set.at[:, jnp.arange(dim) % 2 == 0].set(dim_1_vals)
            self.test_set = test_set
        else:
            raise NotImplementedError("still need to implement this")

        self.shallow_well_bounds = [-1.75, -1.65]
        self.deep_well_bounds = [1.7, 1.8]
        self._plot_bound = 3.

    def log_prob(self, x):
        batched = x.ndim == 2

        if not batched:
            x = x[None,]

        log_probs = jnp.sum(jnp.stack([self.double_well_energy.log_prob(x[..., i * 2:i * 2 + 2]) for
                                       i in range(self.n_wells)], axis=-1), axis=-1, keepdims=True).reshape((-1,))

        if not batched:
            log_probs = jnp.squeeze(log_probs, axis=0)
        return log_probs

    def log_prob_2D(self, x):
        """Marginal 2D pdf - useful for plotting."""
        return self.double_well_energy.log_prob(x)

    def visualise(self, samples: chex.Array = None, axes=None, show=False, prefix='') -> dict:
        """Visualise samples from the model."""
        plotting_bounds = (-3, 3)
        grid_width_n_points = 100
        fig, axs = plt.subplots(2, 2, sharex="row", sharey="row")
        samples = jnp.clip(samples, plotting_bounds[0], plotting_bounds[1])
        for i in range(2):
            for j in range(2):
                # plot contours
                def _log_prob_marginal_pair(x_2d, i, j):
                    x = jnp.zeros((x_2d.shape[0], self.dim))
                    x = x.at[:, i].set(x_2d[:, 0])
                    x = x.at[:, j].set(x_2d[:, 1])
                    return self.log_prob(x)

                xx, yy = jnp.meshgrid(
                    jnp.linspace(plotting_bounds[0], plotting_bounds[1], grid_width_n_points),
                    jnp.linspace(plotting_bounds[0], plotting_bounds[1], grid_width_n_points)
                )
                x_points = jnp.column_stack([xx.ravel(), yy.ravel()])
                log_probs = _log_prob_marginal_pair(x_points, i, j + 2)
                log_probs = jnp.clip(log_probs, -1000, a_max=None).reshape((grid_width_n_points, grid_width_n_points))
                axs[i, j].contour(xx, yy, log_probs, levels=20)

                # plot samples
                axs[i, j].plot(samples[:, i], samples[:, j + 2], "o", alpha=0.5)

                if j == 0:
                    axs[i, j].set_ylabel(f"$x_{i + 1}$")
                if i == 1:
                    axs[i, j].set_xlabel(f"$x_{j + 1 + 2}$")

        wb = {"figures/vis": [wandb.Image(fig)]}
        if show:
            plt.show()
        else:
            plt.close()

        return wb


    def sample(self, seed: chex.PRNGKey, sample_shape: chex.Shape) -> chex.Array:
        return None


if __name__ == '__main__':
    mw = ManyWellEnergy()
    mw.visualise(samples=mw.sample(jax.random.PRNGKey(0), (1,)))

    key = jax.random.PRNGKey(42)
    well = ManyWellEnergy()

    samples = jax.random.normal(key, shape=(10, 32))
    print(samples.shape)
    print((well.log_prob(samples)))
    print((jax.vmap(well.log_prob)(samples)))
