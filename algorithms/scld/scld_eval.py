import jax.numpy as jnp

from algorithms.common.ipm_eval import discrepancies


def eval_scld(simulate_with_smc,
              simulate_without_smc,
              target,
              target_samples,
              config):

    def short_eval(model_state, params, key):
        logger = {}
        model_samples_all, _, (model_lnz_est, model_elbo_est) = simulate_without_smc(key, model_state, params)
        smc_samples_all, _, (smc_lnz_est, smc_elbo_est) = simulate_with_smc(key, model_state, params)

        model_samples = model_samples_all[-1]
        smc_samples = smc_samples_all[-1]

        if target.log_Z is not None:
            logger['metric/model_delta_lnZ'] = jnp.abs(model_lnz_est - target.log_Z)
            logger['metric/smc_delta_lnZ'] = jnp.abs(smc_lnz_est - target.log_Z)
        logger['metric/model_ELBO'] = model_elbo_est
        logger['metric/smc_ELBO'] = smc_elbo_est
        logger['metric/model_target_llh'] = jnp.mean(target.log_prob(smc_samples))
        logger['metric/smc_target_llh'] = jnp.mean(target.log_prob(model_samples))

        if config.compute_emc and config.target.has_entropy:
            logger['metric/model_entropy'] = target.entropy(model_samples)
            logger['metric/smc_entropy'] = target.entropy(smc_samples)

        for d in config.discrepancies:

            logger[f'discrepancies/model_{d}'] = getattr(discrepancies, f'compute_{d}')(target_samples, model_samples,
                                                                                        config) if target_samples is not None else jnp.inf

            logger[f'discrepancies/smc_{d}'] = getattr(discrepancies, f'compute_{d}')(target_samples, smc_samples,
                                                                                      config) if target_samples is not None else jnp.inf

        target.visualise(model_samples, show=config.visualize_samples)
        target.visualise(smc_samples, show=config.visualize_samples)

        if config.algorithm.loss in ['rev_tb', 'fwd_tb']:
            logger['other/mean_log_Z_second_moment'] = jnp.mean(params['params']['logZ'])

        return logger

    return short_eval
