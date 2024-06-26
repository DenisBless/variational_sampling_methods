"""
Code for the General Bridge Sampler (GBS).
Fur further details see: https://arxiv.org/abs/2307.01198
"""
from functools import partial
from time import time

import distrax
import jax
import jax.numpy as jnp
import optax
import wandb
from flax.training import train_state

from algorithms.common.eval_methods.stochastic_oc_methods import get_eval_fn
from algorithms.common.eval_methods.utils import extract_last_entry
from algorithms.gbs.gbs_isw import neg_elbo, rnd
from algorithms.common.models.pisgrad_net import PISGRADNet
from utils.print_util import print_results


def gbs_trainer(cfg, target):
    key_gen = jax.random.PRNGKey(cfg.seed)
    dim = target.dim
    alg_cfg = cfg.algorithm

    # Define prior and target density
    prior = distrax.MultivariateNormalDiag(jnp.zeros(dim),
                                           jnp.ones(dim) * alg_cfg.init_std)
    aux_tuple = (prior.sample, prior.log_prob)

    target_samples = target.sample(jax.random.PRNGKey(0), (cfg.eval_samples,))

    # Define the model
    fwd_model = PISGRADNet(**alg_cfg.model)
    key, key_gen = jax.random.split(key_gen)
    fwd_params = fwd_model.init(key, jnp.ones([alg_cfg.batch_size, dim]),
                                jnp.ones([alg_cfg.batch_size, 1]),
                                jnp.ones([alg_cfg.batch_size, dim]))
    bwd_model = PISGRADNet(**alg_cfg.model)
    key, key_gen = jax.random.split(key_gen)
    bwd_params = bwd_model.init(key, jnp.ones([alg_cfg.batch_size, dim]),
                                jnp.ones([alg_cfg.batch_size, 1]),
                                jnp.ones([alg_cfg.batch_size, dim]))

    optimizer = optax.chain(optax.zero_nans(),
                            optax.clip(alg_cfg.grad_clip),
                            optax.adam(learning_rate=alg_cfg.step_size))
    fwd_state = train_state.TrainState.create(apply_fn=fwd_model.apply, params=fwd_params, tx=optimizer)
    bwd_state = train_state.TrainState.create(apply_fn=bwd_model.apply, params=bwd_params, tx=optimizer)

    loss = jax.jit(jax.grad(neg_elbo, (2, 3), has_aux=True), static_argnums=(4, 5, 6, 7, 8))
    rnd_short = partial(rnd, batch_size=cfg.eval_samples, aux_tuple=aux_tuple,
                        target=target, num_steps=cfg.algorithm.num_steps,
                        noise_schedule=cfg.algorithm.noise_schedule, stop_grad=True)

    eval_fn, logger = get_eval_fn(rnd_short, target, target_samples, cfg)

    eval_freq = max(alg_cfg.iters // cfg.n_evals, 1)
    timer = 0
    for step in range(alg_cfg.iters):
        key, key_gen = jax.random.split(key_gen)
        iter_time = time()

        model_state = (fwd_state, bwd_state)
        (fwd_grads, bwd_grads), _ = loss(key, model_state, fwd_state.params, bwd_state.params, alg_cfg.batch_size,
                                         aux_tuple, target, alg_cfg.num_steps, alg_cfg.noise_schedule)
        timer += time() - iter_time
        fwd_state, bwd_state = model_state

        fwd_state = fwd_state.apply_gradients(grads=fwd_grads)
        bwd_state = bwd_state.apply_gradients(grads=bwd_grads)

        if (step % eval_freq == 0) or (step == alg_cfg.iters - 1):
            key, key_gen = jax.random.split(key_gen)
            logger["stats/step"].append(step)
            logger["stats/wallclock"].append(timer)
            logger["stats/nfe"].append((step + 1) * alg_cfg.batch_size)

            logger.update(eval_fn(model_state, key))
            print_results(step, logger, cfg)

            if cfg.use_wandb:
                wandb.log(extract_last_entry(logger))
