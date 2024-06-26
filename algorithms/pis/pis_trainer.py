"""
Code for Path integral sampler: a stochastic control approach for sampling.
For further details see: https://arxiv.org/abs/2111.15141.
Code based on https://github.com/franciscovargas/denoising_diffusion_samplers
"""

from functools import partial
from time import time

import distrax
import jax
import jax.numpy as jnp
import wandb

from algorithms.common.diffusion_related.init_model import init_model
from algorithms.common.eval_methods.stochastic_oc_methods import get_eval_fn
from algorithms.common.eval_methods.utils import extract_last_entry
from algorithms.pis.pis_rnd import neg_elbo, rnd
from utils.print_util import print_results


def pis_trainer(cfg, target):
    key_gen = jax.random.PRNGKey(cfg.seed)
    dim = target.dim
    alg_cfg = cfg.algorithm

    # Define initial and target density
    normal_log_prob = lambda x, sigma: distrax.MultivariateNormalDiag(jnp.zeros(dim),
                                                                      jnp.ones(dim) * sigma).log_prob(x)
    aux_tuple = (dim, normal_log_prob)
    target_samples = target.sample(jax.random.PRNGKey(0), (cfg.eval_samples,))

    # Initialize the model
    key, key_gen = jax.random.split(key_gen)
    model_state = init_model(key, dim, alg_cfg)

    loss = jax.jit(jax.grad(neg_elbo, 2, has_aux=True), static_argnums=(3, 4, 5, 6, 7))
    rnd_short = partial(rnd, batch_size=cfg.eval_samples, initial_density_tuple=aux_tuple,
                        target=target, num_steps=cfg.algorithm.num_steps,
                        noise_schedule=cfg.algorithm.noise_schedule, stop_grad=True)

    eval_fn, logger = get_eval_fn(rnd_short, target, target_samples, cfg)

    eval_freq = max(alg_cfg.iters // cfg.n_evals, 1)
    timer = 0
    for step in range(alg_cfg.iters):
        key, key_gen = jax.random.split(key_gen)
        iter_time = time()
        grads, _ = loss(key, model_state, model_state.params, alg_cfg.batch_size,
                        aux_tuple, target, alg_cfg.num_steps, alg_cfg.noise_schedule)
        timer += time() - iter_time

        model_state = model_state.apply_gradients(grads=grads)

        if (step % eval_freq == 0) or (step == alg_cfg.iters - 1):
            key, key_gen = jax.random.split(key_gen)
            logger["stats/step"].append(step)
            logger["stats/wallclock"].append(timer)
            logger["stats/nfe"].append((step + 1) * alg_cfg.batch_size)

            logger.update(eval_fn(model_state, key))
            print_results(step, logger, cfg)

            if cfg.use_wandb:
                wandb.log(extract_last_entry(logger))
