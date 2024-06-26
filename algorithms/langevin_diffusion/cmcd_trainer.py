from time import time

import jax
import jax.numpy as jnp
import wandb

from algorithms.langevin_diffusion.ld_eval import eval_langevin
from algorithms.langevin_diffusion.ld_init import initialize_cmcd
from algorithms.langevin_diffusion.ld_utils import collect_eps
from algorithms.langevin_diffusion.optimizer import adam
from algorithms.langevin_diffusion.cmcd import compute_elbo, per_sample_elbo, per_sample_eubo, compute_log_var
from utils.print_util import print_results


def cmcd_trainer(
        cfg,
        target,
        base_dist_params=None,
):
    # Unpack cfg
    dim = target.dim
    alg_cfg = cfg.algorithm
    trainable = alg_cfg.trainable
    lr = alg_cfg.step_size
    iters = alg_cfg.iters
    batch_size = alg_cfg.batch_size
    eval_freq = alg_cfg.iters // 100

    target_log_prob = target.log_prob
    target_samples = target.sample(jax.random.PRNGKey(0), (cfg.eval_samples,))

    params_flat, unflatten, params_fixed = initialize_cmcd(cfg, dim, base_dist_params=base_dist_params)

    evaluate = eval_langevin(per_sample_elbo, per_sample_eubo, unflatten, params_fixed, target, target_samples, cfg)

    if alg_cfg.loss == "var_grad":
        elbo_grad = jax.jit(jax.grad(compute_log_var, 1, has_aux=True), static_argnums=(2, 3, 4, 5, 6))
    elif alg_cfg.loss == "elbo":
        elbo_grad = jax.jit(jax.grad(compute_elbo, 1, has_aux=True), static_argnums=(2, 3, 4, 5, 6))
    else:
        raise NotImplementedError("unknown loss function")

    opt_init, update, get_params = adam(lr)
    update = jax.jit(update, static_argnums=(3, 4))
    opt_state = opt_init(params_flat)
    key, key_gen = jax.random.split(jax.random.PRNGKey(cfg.seed))
    train_losses = []
    test_losses = []
    logger = {"eps": []}

    timer = 0
    for i in range(iters):
        iter_time = time()
        key, key_gen = jax.random.split(key_gen)
        seeds = jax.random.split(key_gen, num=alg_cfg.batch_size)[:, 0]
        params_flat = get_params(opt_state)

        grad, (elbo, x) = elbo_grad(seeds, params_flat, unflatten, params_fixed, target_log_prob, alg_cfg.grad_clipping,
                                    alg_cfg.eps_schedule)

        train_losses.append(jnp.mean(elbo).item())
        if jnp.isnan(jnp.mean(elbo)):
            print("Diverged")
            logger['stats/succ'] = 0
            return [], True, params_flat, logger
        opt_state = update(i, grad, opt_state, unflatten, trainable)
        timer += time() - iter_time

        if (i % eval_freq == 0) or (i == iters - 1):
            key, key_gen = jax.random.split(key_gen)
            logger.update(evaluate(params_flat, key))
            logger["eps"] = collect_eps(params_flat, unflatten, trainable)
            logger["stats/step"] = i
            logger["stats/wallclock"] = timer
            logger["stats/nfe"] = (i + 1) * batch_size * 2 * (alg_cfg.num_temps - 1)

            test_losses.append(logger["metric/ELBO"])

            print_results(i, logger, cfg)

            if cfg.use_wandb:
                wandb.log(logger)

    return (train_losses, test_losses), False, params_flat, logger
