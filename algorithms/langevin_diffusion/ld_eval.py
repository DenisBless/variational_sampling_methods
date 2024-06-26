import jax
import jax.numpy as jnp
import pickle

from jax._src.flatten_util import ravel_pytree

from algorithms.common.eval_methods.utils import moving_averages, save_samples
from algorithms.langevin_diffusion.ld_init import initialize_cmcd, initialize_ldvi, initialize_uha, initialize_mcd, \
    initialize_ula
from algorithms.common.ipm_eval import discrepancies
from utils.path_utils import project_path

from algorithms.langevin_diffusion.cmcd import per_sample_elbo as cmcd_per_sample_elbo
from algorithms.langevin_diffusion.ud_langevin import per_sample_elbo as ud_per_sample_elbo
from algorithms.langevin_diffusion.od_langevin import per_sample_elbo as od_per_sample_elbo


def eval_langevin(per_sample_elbo,
                  per_sample_eubo,
                  unflatten,
                  params_fixed,
                  target,
                  target_samples,
                  cfg):
    target_log_prob = target.log_prob

    logger = {
        'KL/elbo': [],
        'KL/eubo': [],
        'logZ/delta_forward': [],
        'logZ/forward': [],
        'logZ/delta_reverse': [],
        'logZ/reverse': [],
        'ESS/forward': [],
        'ESS/reverse': [],
        'discrepancies/mmd': [],
        'discrepancies/sd': [],
        'other/target_log_prob': [],
        'other/EMC': [],
        "stats/step": [],
        "stats/wallclock": [],
        "stats/nfe": [],
    }

    def short_eval(params_flat, key):
        keys = jax.random.split(key, num=cfg.eval_samples)[:, 0]

        log_ratio, (samples, _) = jax.vmap(per_sample_elbo, in_axes=(0, None, None, None, None))(
            keys, params_flat, unflatten, params_fixed, target_log_prob
        )
        is_weights = jnp.exp(-log_ratio)
        ln_z = jax.scipy.special.logsumexp(-jnp.array(log_ratio)) - jnp.log(cfg.eval_samples)
        elbo = -jnp.mean(log_ratio)

        if target.log_Z is not None:
            logger['logZ/delta_reverse'].append(jnp.abs(ln_z - target.log_Z))

        logger['logZ/reverse'].append(ln_z)
        logger['KL/elbo'].append(elbo)
        logger['ESS/reverse'].append(jnp.sum(is_weights) ** 2 / (cfg.eval_samples * jnp.sum(is_weights ** 2)))
        logger['other/target_log_prob'].append(jnp.mean(target.log_prob(samples)))

        if cfg.compute_forward_metrics and target.can_sample:
            fwd_log_ratio, (fwd_samples, _) = jax.vmap(per_sample_eubo, in_axes=(0, None, None, None, None, 0))(
                keys, params_flat, unflatten, params_fixed, target_log_prob, target_samples
            )

            eubo = -jnp.mean(fwd_log_ratio)
            fwd_ln_z = -(jax.scipy.special.logsumexp(jnp.array(fwd_log_ratio)) - jnp.log(cfg.eval_samples))
            fwd_ess = jnp.exp(
                fwd_ln_z - (jax.scipy.special.logsumexp(-jnp.array(fwd_log_ratio)) - jnp.log(cfg.eval_samples)))

            if target.log_Z is not None:
                logger['logZ/delta_forward'].append(jnp.abs(fwd_ln_z - target.log_Z))
            logger['logZ/forward'].append(fwd_ln_z)
            logger['KL/eubo'].append(eubo)
            logger['ESS/forward'].append(fwd_ess)

        logger.update(target.visualise(samples=samples, show=cfg.visualize_samples))

        if cfg.compute_emc and cfg.target.has_entropy:
            logger['other/EMC'].append(target.entropy(samples))

        for d in cfg.discrepancies:
            logger[f'discrepancies/{d}'].append(getattr(discrepancies, f'compute_{d}')(target_samples, samples,
                                                                                       cfg) if target_samples is not None else jnp.inf)

        if cfg.moving_average.use_ma:
            logger.update(moving_averages(logger, window_size=cfg.moving_average.window_size))

        if cfg.save_samples:
            save_samples(cfg, logger, samples)


        return logger

    return short_eval


def load_and_eval_langevin(model_path,
                           cfg,
                           target,
                           base_dist_params=None, ):
    params_train = pickle.load(open(project_path(f'models/{model_path}_train.pkl'), 'rb'))
    params_notrain = pickle.load(open(project_path(f'models/{model_path}_notrain.pkl'), 'rb'))
    params_flat, unflatten = ravel_pytree((params_train, params_notrain))

    if cfg.alg == 'cmcd':
        _, _, params_fixed = initialize_cmcd(cfg, base_dist_params=base_dist_params)
        per_sample_elbo = cmcd_per_sample_elbo

    elif cfg.alg == 'ula':
        _, _, params_fixed = initialize_ula(cfg, base_dist_params=base_dist_params)
        per_sample_elbo = od_per_sample_elbo

    elif cfg.alg == 'mcd':
        _, _, params_fixed = initialize_mcd(cfg, base_dist_params=base_dist_params)
        per_sample_elbo = od_per_sample_elbo

    elif cfg.alg == 'uha':
        _, _, params_fixed = initialize_uha(cfg, base_dist_params=base_dist_params)
        per_sample_elbo = ud_per_sample_elbo

    elif cfg.alg == 'ldvi':
        _, _, params_fixed = initialize_ldvi(cfg, base_dist_params=base_dist_params)
        per_sample_elbo = ud_per_sample_elbo

    else:
        raise ValueError(f"{cfg.alg} is not a valid algorithm.")

    target_samples = target.sample(jax.random.PRNGKey(0), (cfg.eval_samples,))
    eval = eval_langevin(per_sample_elbo, unflatten, params_fixed, target, target_samples,
                         cfg)  # todo add per sample elbo
    eval(params_flat, jax.random.PRNGKey(cfg.seed))
