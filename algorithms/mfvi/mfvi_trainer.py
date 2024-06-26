"""
Code for Gaussian Mean Field Variational Inference (MFVI).
"""

from time import time
import jax
import jax.numpy as jnp
import wandb
from jax import grad, jit
import optax
import distrax

from algorithms.common.eval_methods.tractable_density_methods import get_eval_fn
from algorithms.common.eval_methods.utils import extract_last_entry
from algorithms.common.utils import get_optimizer
from targets.base_target import Target
from utils.print_util import print_results


# Diagonal Gaussian Variational Distribution
def initialize_variational_params(dim, init_mean, init_diagonal_std):
    initial_mean = jnp.ones(dim) * init_mean
    initial_log_var = jnp.log(jnp.ones(dim) * init_diagonal_std) * 2
    return initial_mean, initial_log_var


# ELBO (Evidence Lower Bound) objective
def neg_elbo(params, key, target_log_density, num_samples):
    mean, log_var = params
    std = jnp.exp(.5 * log_var)
    samples, log_q = distrax.MultivariateNormalDiag(mean, std).sample_and_log_prob(seed=key,
                                                                                   sample_shape=(num_samples,))
    log_p_x = jnp.mean(jax.vmap(target_log_density)(samples))
    elbo = log_p_x - jnp.mean(log_q)
    return -elbo


def sample(params, key, num_samples):
    mean, log_var = params
    std = jnp.exp(.5 * log_var)
    return distrax.MultivariateNormalDiag(mean, std).sample(seed=key, sample_shape=(num_samples,))


# Training loop with optax.adam
def mfvi_trainer(cfg, target: Target):
    @jax.jit
    def rev_log_probs_and_samples(key, params):
        mean, log_var = params
        std = jnp.exp(.5 * log_var)

        key, subkey = jax.random.split(key)
        samples, model_log_prob = distrax.MultivariateNormalDiag(mean, std).sample_and_log_prob(seed=key,
                                                                                                sample_shape=(
                                                                                                    cfg.eval_samples,))
        target_log_prob = jax.vmap(target_log_density)(samples)
        return model_log_prob, target_log_prob, samples

    @jax.jit
    def fwd_log_probs(params):
        mean, log_var = params
        std = jnp.exp(.5 * log_var)
        target_log_p = target_log_density(target_samples)
        model_log_p = distrax.MultivariateNormalDiag(mean, std).log_prob(target_samples)
        return model_log_p, target_log_p

    def eval_mfvi(key):
        model_log_prob, target_log_prob, samples = rev_log_probs_and_samples(key, params)
        if cfg.compute_forward_metrics and (target_samples is not None):
            fwd_model_log_prob, fwd_target_log_p = fwd_log_probs(params)
            logger = eval_fn(samples, target_log_prob - model_log_prob, target_log_prob,
                             fwd_target_log_p - fwd_model_log_prob)
        else:
            logger = eval_fn(samples, target_log_prob - model_log_prob, target_log_prob,
                             None)

        return logger

    dim = target.dim
    alg_cfg = cfg.algorithm
    eval_freq = max(alg_cfg.iters // cfg.n_evals, 1)

    key = jax.random.PRNGKey(cfg.seed)
    params = initialize_variational_params(dim, alg_cfg.init_mean, alg_cfg.init_std)
    optimizer = get_optimizer(alg_cfg.step_size, None)
    opt_state = optimizer.init(params)
    target_log_density = target.log_prob
    target_samples = target.sample(jax.random.PRNGKey(0), (cfg.eval_samples,))

    eval_fn, logger = get_eval_fn(cfg, target, target_samples)

    @jit
    def update(params, opt_state, key):
        gradient = grad(neg_elbo)(params, key, target_log_density, alg_cfg.batch_size)
        updates, new_opt_state = optimizer.update(gradient, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state

    timer = 0
    for step in range(alg_cfg.iters):

        iter_time = time()
        key, subkey = jax.random.split(key)

        params, opt_state = update(params, opt_state, subkey)
        timer += time() - iter_time

        if (step % eval_freq == 0) or (step == alg_cfg.iters - 1):
            key, subkey = jax.random.split(key)
            logger = eval_mfvi(subkey)
            logger["stats/step"].append(step)
            logger["stats/wallclock"].append(timer)
            logger["stats/nfe"].append((step + 1) * alg_cfg.batch_size)

            print_results(step, logger, cfg)

            if cfg.use_wandb:
                wandb.log(extract_last_entry(logger))
