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
from algorithms.common.models.pisgrad_net import PISGRADNet
from algorithms.ldvi.ldvi_isw import neg_elbo, rnd
from utils.helper import inverse_softplus
from utils.print_util import print_results


def ldvi_trainer(cfg, target):
    key_gen = jax.random.PRNGKey(cfg.seed)
    dim = target.dim
    alg_cfg = cfg.algorithm

    # Define initial and target density
    target_samples = target.sample(jax.random.PRNGKey(0), (cfg.eval_samples,))

    # Define the model
    key, key_gen = jax.random.split(key_gen)
    model = PISGRADNet(**alg_cfg.model)
    key, key_gen = jax.random.split(key)
    params = model.init(key, jnp.ones([alg_cfg.batch_size, dim * 2]),
                        jnp.ones([alg_cfg.batch_size, 1]),
                        jnp.ones([alg_cfg.batch_size, dim]))

    additional_params = {'betas': jnp.ones((alg_cfg.num_steps,)),
                         'prior_mean': jnp.zeros((dim,)),
                         'prior_std': jnp.ones((dim,)) * inverse_softplus(alg_cfg.init_std),
                         'diff_coefficient': jnp.ones((1,)) * inverse_softplus(1.),
                         'friction': jnp.ones(1,) * inverse_softplus(alg_cfg.init_momentum)}

    params['params'] = {**params['params'], **additional_params}

    def prior_sampler(params, key, n_samples):
        samples = distrax.MultivariateNormalDiag(params['params']['prior_mean'],
                                                 jnp.ones(dim) * jax.nn.softplus(params['params']['prior_std'])).sample(
            seed=key,
            sample_shape=(
                n_samples,))
        return samples if alg_cfg.learn_prior else jax.lax.stop_gradient(samples)

    if alg_cfg.learn_prior:
        def prior_log_prob(params, x):
            log_probs = distrax.MultivariateNormalDiag(params['params']['prior_mean'],
                                                       jnp.ones(dim) * jax.nn.softplus(
                                                           params['params']['prior_std'])).log_prob(x)
            return log_probs
    else:
        def prior_log_prob(params, x):
            log_probs = distrax.MultivariateNormalDiag(jnp.zeros(dim),
                                                       jnp.ones(dim) * alg_cfg.init_std).log_prob(x)
            return log_probs

    def get_betas(params):
        b = jax.nn.softplus(params['params']['betas'])
        b = jnp.cumsum(b) / jnp.sum(b)
        b = b if alg_cfg.learn_betas else jax.lax.stop_gradient(b)

        def get_beta(step):
            return b[jnp.array(step, int)]

        return get_beta

    def get_diff_coefficient(params):
        diff_coefficient = jax.nn.softplus(params['params']['diff_coefficient'])
        return diff_coefficient if alg_cfg.learn_diffusion_coefficient else jax.lax.stop_gradient(diff_coefficient)

    def get_friction(params):
        friction = jax.nn.softplus(params['params']['friction'])
        return friction if alg_cfg.learn_friction else jax.lax.stop_gradient(friction)

    aux_tuple = (prior_sampler, prior_log_prob, get_betas, get_diff_coefficient, get_friction)

    optimizer = optax.chain(optax.zero_nans(),
                            optax.clip(alg_cfg.grad_clip),
                            optax.adam(learning_rate=alg_cfg.step_size))

    model_state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

    loss = jax.jit(jax.grad(neg_elbo, 2, has_aux=True), static_argnums=(3, 4, 5, 6, 7))
    rnd_short = partial(rnd, batch_size=cfg.eval_samples, aux_tuple=aux_tuple,
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
