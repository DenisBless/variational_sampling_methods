import jax.numpy as jnp

from algorithms.common.ipm_eval import mmd_median, optimal_transport


def compute_mmd(gt_samples, samples, config):
    return getattr(mmd_median, 'mmd_median')(gt_samples, samples) if gt_samples is not None else jnp.inf


def compute_sd(gt_samples, samples, config):
    return getattr(optimal_transport, 'SD')(gt_samples).compute_SD(samples) if gt_samples is not None else jnp.inf


def compute_eot(gt_samples, samples, config):
    return getattr(optimal_transport, 'OT')(gt_samples).compute_OT(samples) if gt_samples is not None else jnp.inf
