import os

import jax.numpy as jnp
from typing import List
import jax
import chex
import distrax
import wandb

from targets.base_target import Target
from utils.path_utils import project_path

import numpy as np
import matplotlib.pyplot as plt


def visualize_samples(samples, show=False):
    def visualize_n_link(theta, num_dimensions, l):
        x = [0]
        y = [0]
        for i in range(0, num_dimensions):
            y.append(y[-1] + l[i] * np.sin(np.sum(theta[:i + 1])))
            x.append(x[-1] + l[i] * np.cos(np.sum(theta[:i + 1])))
            ax.plot([x[-2], x[-1]], [y[-2], y[-1]], color='k', linestyle='-', linewidth=2, alpha=0.3)
        ax.plot(x[-1], y[-1], 'o', c='k')
        ax.plot(0.7 * num_dimensions, 0, 'rx')
        return ax

    fig, ax = plt.subplots()
    num_dimensions = samples.shape[1]
    ax.set_xlim([-0.2 * num_dimensions, num_dimensions])
    ax.set_ylim([-0.5 * num_dimensions, 0.5 * num_dimensions])
    [num_samples, num_dimensions] = samples.shape
    for i in range(0, num_samples):
        visualize_n_link(samples[i], num_dimensions, np.ones(num_dimensions))

    wb = {"figures/vis": [wandb.Image(fig)]}
    if show:
        plt.show()
    return wb

    # import tikzplotlib
    # tikzplotlib.save(os.path.join(project_path('./figures/'), f"robot.tex"))


class PlanarRobot(Target):
    def __init__(self, dim: int, num_goals: int = 1, prior_std=2e-1, likelihood_std=1e-2, log_Z=None,
                 can_sample=False, sample_bounds=None):
        super().__init__(dim, log_Z, can_sample)
        self.num_links = dim
        prior_stds = jnp.full((dim,), prior_std)
        prior_stds = prior_stds.at[0].set(1.)
        self.prior = distrax.MultivariateNormalDiag(loc=jnp.zeros(dim), scale_diag=prior_stds)
        self.link_lengths = jnp.ones(self.dim)

        # Load ground truth samples
        self.gt_samples = jnp.array(jnp.load(project_path('targets/data/planar_robot_gt_10k.npz'))['arr_0'])
        self.num_gt_samples = self.gt_samples.shape[0]

        if num_goals == 1:
            self.goals = jnp.array([[7.0, 0.0]], dtype=jnp.float32)
        elif num_goals == 4:
            self.goals = jnp.array([[7.0, 0.0], [-7.0, 0.0], [0.0, 7.0], [0.0, -7.0]], dtype=jnp.float32)
        else:
            raise ValueError("Number of goals must be 1 or 4")

        self.goal_Gaussians = []
        for goal in self.goals:
            goal_std = jnp.ones(2) * likelihood_std
            self.goal_Gaussians.append(distrax.MultivariateNormalDiag(loc=goal, scale_diag=goal_std))

    def likelihood(self, pos):
        likelihoods = jnp.stack([goal.log_prob(pos) for goal in self.goal_Gaussians], axis=0)
        return jnp.max(likelihoods, axis=0)

    # def forward_kinematics(self, theta):
    #     y = 0.
    #     x = 0.
    #     for i in range(self.dim):
    #         y += self.link_lengths[i] * jnp.sin(jnp.sum(theta[:i + 1]))
    #         x += self.link_lengths[i] * jnp.cos(jnp.sum(theta[:i + 1]))
    #     return jnp.column_stack((x, y))
    def forward_kinematics(self, theta):  # todo implement the batched version from oleg and follow the other target functions
        y = jnp.zeros(theta.shape[0])
        x = jnp.zeros(theta.shape[0])
        for i in range(self.dim):
            y += self.link_lengths[i] * jnp.sin(jnp.sum(theta[:, :i + 1], axis=1))
            x += self.link_lengths[i] * jnp.cos(jnp.sum(theta[:, :i + 1], axis=1))
        return jnp.stack((x, y), axis=1)

    def log_prob(self, theta):
        batched = theta.ndim == 2

        if not batched:
            theta = theta[None,]

        log_prob = self.prior.log_prob(theta) + self.likelihood(self.forward_kinematics(theta))

        if not batched:
            log_prob = jnp.squeeze(log_prob, axis=0)

        return log_prob

        # per_sample_lp = lambda x: self.prior.log_prob(x) + self.likelihood(self.forward_kinematics(x))
        # lps = jax.vmap(per_sample_lp)(theta).reshape(-1, )
        # return self.prior.log_prob(theta) + self.likelihood(self.forward_kinematics(theta))
        # return self.likelihood(self.forward_kinematics(theta))
        # return lps

    def visualise(self, samples: chex.Array = None, axes=None, show=False, prefix='') -> dict:
        """Visualise samples from the model."""
        plt.close()
        num_samples = 100
        idx = jax.random.choice(jax.random.PRNGKey(0), samples.shape[0], shape=(num_samples,))
        return visualize_samples(samples[idx], show=show)

    def sample(self, seed: chex.PRNGKey, sample_shape: chex.Shape) -> chex.Array:
        # Generate a subset of the ground truth sample set
        indices = jax.random.choice(seed, self.num_gt_samples, shape=sample_shape, replace=False)
        # Use the generated indices to select the subset
        return self.gt_samples[indices]


if __name__ == '__main__':
    pr = PlanarRobot(dim=10)
    samples = pr.sample(jax.random.PRNGKey(0), (2000,))
    pr.visualise(samples, show=True)
