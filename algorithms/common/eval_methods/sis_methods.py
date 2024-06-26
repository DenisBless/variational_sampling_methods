import jax.numpy as jnp

from algorithms.common.eval_methods.utils import moving_averages, save_samples
from algorithms.common.ipm_eval import discrepancies


def get_eval_fn(cfg, target, target_samples):
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

    n_eval_samples = cfg.eval_samples

    def eval_fn(samples, elbo, rev_lnz, eubo, fwd_lnz):

        if target.log_Z is not None:
            logger['logZ/delta_reverse'].append(jnp.abs(rev_lnz - target.log_Z))

        logger['logZ/reverse'].append(rev_lnz)
        logger['KL/elbo'].append(elbo)
        logger['other/target_log_prob'].append(jnp.mean(target.log_prob(samples)))

        if cfg.compute_forward_metrics and (target_samples is not None):
            if target.log_Z is not None:
                logger['logZ/delta_forward'].append(jnp.abs(fwd_lnz - target.log_Z))
            logger['logZ/forward'].append(fwd_lnz)
            logger['KL/eubo'].append(eubo)

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

    return eval_fn, logger
