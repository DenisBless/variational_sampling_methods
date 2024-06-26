"""Code builds on https://github.com/lollcat/fab-jax"""
from typing import Union
import jax
import jax.numpy as jnp
from algorithms.fab.sampling.smc import SequentialMonteCarloSampler
from algorithms.fab.flow.flow import Flow
from algorithms.fab.train.fab_with_buffer import TrainStateWithBuffer
from algorithms.fab.train.fab_without_buffer import TrainStateNoBuffer
import chex
from typing import Optional
from targets.base_target import Target


# max_elbo = -10000000000000


def setup_fab_eval_function(flow: Flow, ais: SequentialMonteCarloSampler, target: Target, config):
    assert ais.alpha == 1.  # Make sure target is set to p.
    assert ais.use_resampling is False  # Make sure we are doing AIS, not SMC.

    # def eval_fn(state: Union[TrainStateNoBuffer, TrainStateWithBuffer], key: chex.PRNGKey):
    #     def log_q_fn(x: chex.Array) -> chex.Array:
    #         return flow.log_prob_apply(state.flow_params, x)
    #
    #     key1, key2 = jax.random.split(key)
    #
    #     # Perform SMC forward pass and grab just the importance weights.
    #     x0, log_q_flow = flow.sample_and_log_prob_apply(state.flow_params, key1, (config.eval_samples,))
    #     log_w_flow = target.log_prob(x0) - log_q_flow
    #     ais_point, log_w_ais, _, _ = ais.step(x0, state.smc_state, log_q_fn, target.log_prob)
    #     ais_samples = ais_point.x
    #
    #     # compute metrics
    #     log_z_flow = jax.nn.logsumexp(log_w_flow, axis=-1) - jnp.log(config.eval_samples)
    #     log_z_ais = jax.nn.logsumexp(log_w_ais, axis=-1) - jnp.log(config.eval_samples)
    #     elbo_flow = jnp.mean(log_w_flow)
    #     elbo_ais = jnp.mean(log_w_ais)
    #     logger = {}
    #
    #     logger["metric/lnZ"] = log_z_ais
    #     if target.log_Z is not None:
    #         logger['metric/delta_lnZ'] = jnp.abs(log_z_ais - target.log_Z)
    #     else:
    #         logger['metric/lnZ'] = log_z_ais
    #     logger["metric/ELBO"] = elbo_ais
    #     logger['metric/target_llh'] = jnp.mean(target.log_prob(ais_samples))
    #     logger["metric/reverse_ESS"] = jnp.exp(log_effective_sample_size(log_w_ais))
    #
    #     if target.log_Z is not None:
    #         logger["mean_abs_err_log_z"] = jnp.mean(jnp.abs(log_z_ais - target.log_Z))
    #
    #     if config.compute_emc and config.target.has_entropy:
    #         logger['metric/entropy'] = target.entropy(ais_samples)
    #
    #     target_samples = target.sample(jax.random.PRNGKey(0), (config.eval_samples,))
    #     for d in config.discrepancies:
    #         logger[f'discrepancies/{d}'] = (getattr(discrepancies, f'compute_{d}')
    #                                         (target_samples, ais_samples,
    #                                          config)) if target_samples is not None else jnp.inf
    #
    #     if target.can_sample:
    #         logger["forward_ESS"] = calculate_log_forward_ess(log_w_ais, log_Z=target.log_Z)
    #
    #     # global max_elbo
    #     # print("fab max elbo: ", max_elbo)
    #     # if elbo_ais.item() > max_elbo:
    #     #     max_elbo = elbo_ais.item()
    #     #     jnp.save(project_path(f'samples/fab_robot_samples'), ais_samples)
    #     #
    #     # return logger
    #
    #     # Reverse process
    #     if config.compute_forward_metrics and (target_samples is not None):
    #         reverse_ais_point, reverse_log_w_ais, _, _ = ais.reverse_step(target_samples, state.smc_state, log_q_fn,
    #                                                                       target.log_prob)
    #
    #         eubo_ais = jnp.mean(reverse_log_w_ais)
    #         reverse_log_z_ais = -(jax.nn.logsumexp(reverse_log_w_ais, axis=-1) - jnp.log(config.eval_samples))
    #         logger['metric/EUBO'] = eubo_ais
    #         logger['metric/rev_lnZ'] = reverse_log_z_ais
    #         # reverse_ais_samples = reverse_ais_point.x
    #
    #     return logger
    def eval_fn(state: Union[TrainStateNoBuffer, TrainStateWithBuffer], key: chex.PRNGKey):
        def log_q_fn(x: chex.Array) -> chex.Array:
            return flow.log_prob_apply(state.flow_params, x)

        key1, key2 = jax.random.split(key)

        target_samples = target.sample(jax.random.PRNGKey(0), (config.eval_samples,))
        # Perform SMC forward pass and grab just the importance weights.
        x0, log_q_flow = flow.sample_and_log_prob_apply(state.flow_params, key1, (config.eval_samples,))
        log_w_flow = target.log_prob(x0) - log_q_flow
        ais_point, log_w_ais, _, _ = ais.step(x0, state.smc_state, log_q_fn, target.log_prob)
        ais_samples = ais_point.x

        # compute metrics
        log_z_ais = jax.nn.logsumexp(log_w_ais, axis=-1) - jnp.log(config.eval_samples)
        elbo_ais = jnp.mean(log_w_ais)

        # Reverse process
        if config.compute_forward_metrics and (target_samples is not None):
            fwd_ais_point, fwd_log_w_ais, _, _ = ais.reverse_step(target_samples, state.smc_state, log_q_fn,
                                                                  target.log_prob)

            eubo_ais = jnp.mean(fwd_log_w_ais)
            fwd_log_z_ais = -(jax.nn.logsumexp(fwd_log_w_ais, axis=-1) - jnp.log(config.eval_samples))

            return ais_samples, elbo_ais, log_z_ais, eubo_ais, fwd_log_z_ais

        else:
            return ais_samples, elbo_ais, log_z_ais, None, None

    return eval_fn


def calculate_log_forward_ess(
        log_w: chex.Array,
        mask: Optional[chex.Array] = None,
        log_Z: Optional[float] = None
) -> chex.Array:
    """Calculate forward ess, either using exact log_Z if it is known, or via estimating it from the samples.
    NB: log_q = p(x)/q(x) where x ~ p(x).
    """
    if mask is None:
        mask = jnp.ones_like(log_w)

    chex.assert_equal_shape((log_w, mask))
    log_w = jnp.where(mask, log_w, jnp.zeros_like(log_w))  # make sure log_w finite

    if log_Z is None:
        log_z_inv = jax.nn.logsumexp(-log_w, b=mask) - jnp.log(jnp.sum(mask))
    else:
        log_z_inv = - log_Z

    # log ( Z * E_p[p(x)/q(x)] )
    log_z_times_expectation_p_of_p_div_q = jax.nn.logsumexp(log_w, b=mask) - jnp.log(jnp.sum(mask))
    # ESS (as fraction of 1) = 1/E_p[p(x)/q(x)]
    # ESS = Z / ( Z * E_p[p(x)/q(x)] )
    # Log ESS = - log Z^{-1} -  log ( Z * E_p[p(x)/q(x)] )
    log_forward_ess = - log_z_inv - log_z_times_expectation_p_of_p_div_q
    return log_forward_ess
