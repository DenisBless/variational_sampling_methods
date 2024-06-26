import itertools
import os
from collections.abc import MutableMapping
from typing import Optional

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import ml_collections
import numpy as np
import wandb
from chex import Array
# from configs.base import FUNNEL_EPS_DICT, LR_DICT
from jax import scipy as jscipy


def make_grid(x: Array, im_size: int, n: int = 16, wandb_prefix: str = ""):
    """
    Plot a grid of images, and optionally log to wandb.

    x: (N, im_size, im_size) array of images
    im_size: size of images
    n: number of images to plot
    wandb_prefix: prefix to use for wandb logging
    """
    x = np.array(x[:n].reshape(-1, im_size, im_size))

    n_rows = int(np.sqrt(n))
    fig, ax = plt.subplots(n_rows, n_rows, figsize=(8, 8))

    # Plot each image
    for i in range(n_rows):
        for j in range(n_rows):
            ax[i, j].imshow(x[i * n_rows + j], cmap="gray")
            ax[i, j].axis("off")

    # Log into wandb
    wandb.log({f"{wandb_prefix}": fig})
    plt.close()


# Taken from https://github.com/lollcat/fab-jax/blob/632e0a7d3dbd8da6b2ef043ab41e2346f29dfece/fabjax/utils/plot.py#L11
def plot_contours_2D(
    log_prob_func, ax: Optional[plt.Axes] = None, bound: int = 3, levels: int = 20
):
    """Plot the contours of a 2D log prob function."""
    if ax is None:
        fig, ax = plt.subplots(1)
    n_points = 200
    x_points_dim1 = np.linspace(-bound, bound, n_points)
    x_points_dim2 = np.linspace(-bound, bound, n_points)
    x_points = np.array(list(itertools.product(x_points_dim1, x_points_dim2)))
    log_probs = log_prob_func(x_points)
    log_probs = jnp.clip(log_probs, a_min=-1000, a_max=None)
    x1 = x_points[:, 0].reshape(n_points, n_points)
    x2 = x_points[:, 1].reshape(n_points, n_points)
    z = log_probs.reshape(n_points, n_points)
    ax.contour(x1, x2, z, levels=levels)


# Taken from https://github.com/lollcat/fab-jax/blob/632e0a7d3dbd8da6b2ef043ab41e2346f29dfece/fabjax/utils/plot.py#L30
def plot_marginal_pair(
    samples, ax=None, marginal_dims=(0, 1), bounds=(-5, 5), alpha: float = 0.5
):
    """Plot samples from marginal of distribution for a given pair of dimensions."""
    if not ax:
        fig, ax = plt.subplots(1)
    samples = jnp.clip(samples, bounds[0], bounds[1])
    ax.plot(
        samples[:, marginal_dims[0]], samples[:, marginal_dims[1]], "o", alpha=alpha
    )


def plot_gmm(samples, log_p_fn, loc_scaling, wandb_prefix: str = ""):
    plot_bound = loc_scaling * 1.5
    fig, axs = plt.subplots(1, figsize=(5, 5))
    plot_marginal_pair(samples, axs, bounds=(-plot_bound, plot_bound))
    plot_contours_2D(log_p_fn, axs, bound=plot_bound, levels=50)
    axs.set_title("samples")
    plt.tight_layout()
    wandb.log({f"{wandb_prefix}": wandb.Image(fig)})
    plt.close()


# Taken from https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys
def flatten_nested_dict(nested_dict, parent_key="", sep="."):
    items = []
    for name, cfg in nested_dict.items():
        new_key = parent_key + sep + name if parent_key else name
        if isinstance(cfg, MutableMapping):
            items.extend(flatten_nested_dict(cfg, new_key, sep=sep).items())
        else:
            items.append((new_key, cfg))

    return dict(items)


def update_config_dict(config_dict: ml_collections.ConfigDict, run, new_vals: dict):
    config_dict.unlock()
    config_dict.update_from_flattened_dict(run.config)
    config_dict.update_from_flattened_dict(new_vals)
    run.config.update(new_vals, allow_val_change=True)
    config_dict.lock()


def setup_training(wandb_run):
    """Helper function that sets up training configs and logs to wandb."""
    if not wandb_run.config.get("use_tpu", False):
        # # TF can hog GPU memory, so we hide the GPU device from it.
        # tf.config.experimental.set_visible_devices([], "GPU")

        # Without this, JAX is automatically using 90% GPU for pre-allocation.
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.3"
        # os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false"
        # Disable logging of compiles.
        jax.config.update("jax_log_compiles", False)

        # Log various JAX configs to wandb, and locally.
        wandb_run.summary.update(
            {
                "jax_process_index": jax.process_index(),
                "jax.process_count": jax.process_count(),
            }
        )
    else:
        # config.FLAGS.jax_xla_backend = "tpu_driver"
        # config.FLAGS.jax_backend_target = os.environ['TPU_NAME']
        # DEVICE_COUNT = len(jax.local_devices())
        print(jax.default_backend())
        print(jax.device_count(), jax.local_device_count())
        print("8 cores of TPU ( Local devices in Jax ):")
        print("\n".join(map(str, jax.local_devices())))


# def plot_samples(
#     model,
#     log_prob_model,
#     samples,
#     info,
#     target_samples=None,
#     log_prefix=None,
#     ema_samples=None,
# ):
#     if model == "nice":
#         make_grid(samples, info.im_size, n=64, wandb_prefix=f"{log_prefix}/images")
#     if model == "many_gmm":
#         plot_gmm(
#             samples,
#             log_prob_model,
#             info.loc_scaling,
#             wandb_prefix=f"{log_prefix}/samples",
#         )
#     if ema_samples is not None:
#         if model == "nice":
#             make_grid(
#                 ema_samples, info.im_size, n=64, wandb_prefix=f"{log_prefix}/ema_images"
#             )
#     if target_samples is not None:
#         if model == "nice":
#             make_grid(
#                 target_samples, info.im_size, n=64, wandb_prefix=f"{log_prefix}/target"
#             )
#         if model == "many_gmm":
#             plot_gmm(
#                 target_samples,
#                 log_prob_model,
#                 info.loc_scaling,
#                 wandb_prefix=f"{log_prefix}/target",
#             )
#         wandb.log(
#             {
#                 f"{log_prefix}/w2": W2_distance(
#                     samples[: info.n_sinkhorn, ...],
#                     target_samples[: info.n_sinkhorn, ...],
#                 )
#             }
#         )


# def setup_config(wandb_config, config):
#     try:
#         if wandb_config.model == "nice":
#             config.model = (
#                 wandb_config.model
#                 + f"_{wandb_config.alpha}_{wandb_config.n_bits}_{wandb_config.im_size}"
#             )
#             new_vals = {}
#         elif wandb_config.model in ["funnel"]:
#             values = FUNNEL_EPS_DICT[wandb_config.nbridges]
#             new_vals = {"init_eps": values["init_eps"], "lr": values["lr"]}
#         elif wandb_config.model in ["many_gmm", "gmm"]:
#             new_vals = {}
#         else:
#             new_vals = {"lr": LR_DICT[wandb_config.model][wandb_config.boundmode]}
#             print(new_vals)
#     except KeyError:
#         new_vals = {}
#         print(
#             "LR not found for model %s and boundmode %s"
#             % (wandb_config.model, wandb_config.boundmode)
#         )
#
#     return new_vals



def log_final_losses(eval_losses, log_prefix=""):
    """
    eval_losses is of shape (n_input_dist_seeds, n_samples)
    """
    # (n_input_dist_seeds, n_samples)
    eval_losses = jnp.array(eval_losses)
    n_samples = eval_losses.shape[1]
    # Calculate mean and std of ELBOs over 30 seeds
    final_elbos = -jnp.mean(eval_losses, axis=1)
    final_elbo = jnp.mean(final_elbos)
    final_elbo_std = jnp.std(final_elbos)

    # Calculate mean and std of log Zs over 30 seeds
    ln_numsamp = jnp.log(n_samples)

    final_ln_Zs = jscipy.special.logsumexp(-jnp.array(eval_losses), axis=1) - ln_numsamp

    final_ln_Z = jnp.mean(final_ln_Zs)
    final_ln_Z_std = jnp.std(final_ln_Zs)

    wandb.log(
        {
            f"elbo_final{log_prefix}": np.array(final_elbo),
            f"final_ln_Z{log_prefix}": np.array(final_ln_Z),
            f"elbo_final_std{log_prefix}": np.array(final_elbo_std),
            f"final_ln_Z_std{log_prefix}": np.array(final_ln_Z_std),
        }
    )

    return final_elbo, final_ln_Z
