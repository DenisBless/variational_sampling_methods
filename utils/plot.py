from typing import Optional, Tuple
import itertools

import distrax
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import chex
import pandas as pd
import wandb


def plot_gaussian_contours_2D(std,
                              samples,
                              ax: Optional[plt.Axes] = None,
                              levels: int = 20,
                              show=False):
    """Plot the contours of a 2D log prob function."""
    n_points = 100
    dim = samples.shape[1]
    if ax is None:
        fig, ax = plt.subplots(1)
    x_points_dim1 = np.linspace(-4 * std, 4 * std, n_points)
    x_points_dim2 = np.linspace(-4 * std, 4 * std, n_points)
    x_points = np.array(list(itertools.product(x_points_dim1, x_points_dim2)))
    log_prob_func = distrax.MultivariateNormalDiag(loc=jnp.zeros(2), scale_diag=jnp.ones(2) * std).log_prob
    log_probs = log_prob_func(x_points)
    log_probs = jnp.clip(log_probs, a_min=-1000, a_max=None)
    x1 = x_points[:, 0].reshape(n_points, n_points)
    x2 = x_points[:, 1].reshape(n_points, n_points)
    z = log_probs.reshape(n_points, n_points)
    ax.contourf(x1, x2, jnp.exp(z), levels=levels)

    ax.scatter(samples[:, 0], samples[:, 1], c='r', alpha=0.6, marker='x')

    try:
        wandb.log({"images/backward": wandb.Image(plt)})
    except:
        pass

    if show:
        plt.show()


def plot_history(history):
    """Agnostic history plotter for quickly plotting a dictionary of logging info."""
    figure, axs = plt.subplots(len(history), 1, figsize=(7, 3 * len(history.keys())))
    if len(history.keys()) == 1:
        axs = [axs]  # make iterable
    elif len(history.keys()) == 0:
        return
    for i, key in enumerate(history):
        data = pd.Series(history[key])
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        if sum(data.isna()) > 0:
            data = data.dropna()
            print(f"NaN encountered in {key} history")
        axs[i].plot(data)
        axs[i].set_title(key)
    plt.tight_layout()
