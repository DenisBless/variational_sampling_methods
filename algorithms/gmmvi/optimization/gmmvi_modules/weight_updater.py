from typing import NamedTuple, Callable
from jax import lax
from algorithms.gmmvi.gmm_vi_utils.utils import reduce_weighted_logsumexp
from algorithms.gmmvi.models.gmm_wrapper import GMMWrapperState, GMMWrapper
import chex
import jax.numpy as jnp
import jax


class WeightUpdaterState(NamedTuple):
    pass


class WeightUpdater(NamedTuple):
    init_weight_updater_state: Callable
    update_weights: Callable


def setup_get_expected_log_ratios(log_densities_also_individual_fn: Callable, store_rewards_fn: Callable,
                                  TEMPERATURE: float, USE_SELF_NORMALIZED_IMPORTANCE_WEIGHTS):
    def get_expected_log_ratios(gmm_wrapper_state: GMMWrapperState, samples, background_mixture_densities, target_lnpdfs):

        model_densities, component_log_densities = jax.vmap(log_densities_also_individual_fn, in_axes=(None, 0))(gmm_wrapper_state.gmm_state, samples)
        component_log_densities = jnp.transpose(component_log_densities)  # transpose because of switched shape
        log_ratios = target_lnpdfs - TEMPERATURE * model_densities
        if USE_SELF_NORMALIZED_IMPORTANCE_WEIGHTS:
            log_weights = component_log_densities - background_mixture_densities
            log_weights -= jax.nn.logsumexp(log_weights, axis=1, keepdims=True)
            weights = jnp.exp(log_weights)
            importance_weights = weights / jnp.sum(weights, axis=1, keepdims=True)
            expected_log_ratios = jnp.dot(importance_weights, log_ratios)
        else:
            n = jnp.array(jnp.shape(samples)[0], jnp.float32)
            log_importance_weights = component_log_densities - background_mixture_densities
            lswe, signs = reduce_weighted_logsumexp(
                log_importance_weights + jnp.log(jnp.abs(log_ratios)),
                w=jnp.sign(log_ratios), axis=1, return_sign=True)
            expected_log_ratios = 1 / n * signs * jnp.exp(lswe)

        component_rewards = TEMPERATURE * gmm_wrapper_state.gmm_state.log_weights + expected_log_ratios
        gmm_wrapper_state = store_rewards_fn(gmm_wrapper_state, component_rewards)
        return gmm_wrapper_state, expected_log_ratios

    return get_expected_log_ratios


def setup_update_weights_fn(get_expected_log_ratios_fn: Callable, update_weights_from_expected_log_ratios_fn: Callable):

    @jax.jit
    def update_weights(gmm_wrapper_state: GMMWrapperState, samples: chex.Array, background_mixture_densities: chex.Array,
                       target_lnpdfs: chex.Array, stepsize: float):
        gmm_wrapper_state, expected_log_ratios = get_expected_log_ratios_fn(gmm_wrapper_state, samples, background_mixture_densities, target_lnpdfs)
        gmm_wrapper_state = update_weights_from_expected_log_ratios_fn(gmm_wrapper_state, expected_log_ratios, stepsize)
        return gmm_wrapper_state

    return update_weights


def setup_direct_weight_updater(gmm_wrapper: GMMWrapper, TEMPERATURE, USE_SELF_NORMALIZED_IMPORTANCE_WEIGHTS):
    def init_direct_weight_updater(temperature: float, use_self_normalized_importance_weights: bool):
        return WeightUpdaterState()

    def _update_weights_from_expected_log_ratios(gmm_wrapper_state: GMMWrapperState,
                                                 expected_log_ratios: chex.Array, stepsize: jnp.float32):
        def true_fn(gmm_wrapper_state, stepsize, expected_log_ratios):
            unnormalized_weights = gmm_wrapper_state.gmm_state.log_weights + stepsize / TEMPERATURE * expected_log_ratios
            new_log_probs = unnormalized_weights - jax.nn.logsumexp(unnormalized_weights)
            new_log_probs = jnp.maximum(new_log_probs, -69.07)  # lower bound weights to 1e-30
            new_log_probs -= jax.nn.logsumexp(new_log_probs)
            return gmm_wrapper.replace_weights(gmm_wrapper_state, new_log_probs)

        return jax.lax.cond(gmm_wrapper_state.gmm_state.num_components > 1,
                            true_fn,
                            lambda gmm_wrapper_state, stepsize, expected_log_ratios: gmm_wrapper_state,
                            gmm_wrapper_state, stepsize, expected_log_ratios)

    get_expected_log_ratios_fn = setup_get_expected_log_ratios(gmm_wrapper.log_densities_also_individual,
                                                               gmm_wrapper.store_rewards, TEMPERATURE, USE_SELF_NORMALIZED_IMPORTANCE_WEIGHTS)
    return WeightUpdater(init_weight_updater_state=init_direct_weight_updater,
                         update_weights=setup_update_weights_fn(get_expected_log_ratios_fn, _update_weights_from_expected_log_ratios))


def setup_trust_region_based_weight_updater(gmm_wrapper: GMMWrapper, TEMPERATURE: float,
                                            USE_SELF_NORMALIZED_IMPORTANCE_WEIGHTS: bool):

    def init_trust_region_based_weight_updater(temperature: float, use_self_normalized_importance_weights: bool):
        return WeightUpdaterState()

    def _kl(gmm_wrapper_state: GMMWrapperState, eta: jnp.float32,
            component_rewards: chex.Array) -> [jnp.float32, chex.Array]:
        unnormalized_weights = ((eta + 1) / (TEMPERATURE + eta) *
                                gmm_wrapper_state.gmm_state.log_weights + 1. / (TEMPERATURE + eta) * component_rewards)
        new_log_weights = unnormalized_weights - jax.nn.logsumexp(unnormalized_weights)
        new_log_weights = jnp.maximum(new_log_weights, -69.07)  # lower bound weights to 1e-30
        new_log_weights -= jax.nn.logsumexp(new_log_weights)

        kl = jnp.sum(jnp.exp(new_log_weights) * (new_log_weights - gmm_wrapper_state.gmm_state.log_weights))
        return kl, new_log_weights

    def _update_weights_from_expected_log_ratios(gmm_wrapper_state: GMMWrapperState, expected_log_ratios, kl_bound):

        def _bracketing_search(gmm_wrapper_state, expected_log_ratios, KL_BOUND):
            lower_bound = jnp.array(-45.)
            upper_bound = jnp.array(45.)
            log_eta = 0.5 * (upper_bound + lower_bound)
            kl = -1
            upper_bound_satisfies_constraint = False
            new_log_weights = gmm_wrapper_state.gmm_state.log_weights
            carry = (0, lower_bound, upper_bound, log_eta, kl, upper_bound_satisfies_constraint, new_log_weights)

            def cond_fn(carry):
                it, lower_bound, upper_bound, _, kl, _, _ = carry
                diff = jnp.abs(jnp.exp(upper_bound) - jnp.exp(lower_bound))
                return (it < 50) & (diff >= 1e-1) & (jnp.abs(KL_BOUND - kl) >= 1e-1 * KL_BOUND)

            def body_fn(carry):
                it, lower_bound, upper_bound, log_eta, _, upper_bound_satisfies_constraint, _ = carry
                eta = jnp.exp(log_eta)
                kl, new_log_weights = _kl(gmm_wrapper_state, eta, expected_log_ratios)

                def true_fn():
                    new_lower_bound = upper_bound
                    return it+1, new_lower_bound, upper_bound, log_eta, kl, upper_bound_satisfies_constraint, new_log_weights

                def false_fn():
                    new_lower_bound, new_upper_bound, new_upper_bound_satisfies_constrain = jax.lax.cond(KL_BOUND > kl,
                                                                                                         lambda lower_bound, upper_bound, log_eta: (lower_bound, log_eta, True),
                                                                                                         lambda lower_bound, upper_bound, log_eta: (log_eta, upper_bound, upper_bound_satisfies_constraint),
                                                                                                         lower_bound, upper_bound, log_eta)
                    new_log_eta = 0.5 * (new_upper_bound + new_lower_bound)
                    return it+1, new_lower_bound, new_upper_bound, new_log_eta, kl, new_upper_bound_satisfies_constrain, new_log_weights

                return jax.lax.cond(jnp.abs(KL_BOUND - kl) < 1e-1*KL_BOUND, true_fn, false_fn)

            _, lower_bound, upper_bound, eta, kl, upper_bound_satisfies_constraint, new_log_weights = jax.lax.while_loop(cond_fn, body_fn, init_val=carry)

            def get_return_values_fn():
                # We could not find the optimal multiplier, but if the upper bound is large enough to satisfy the constraint,
                # we can still make an update
                kl, new_log_weights = _kl(gmm_wrapper_state, jnp.exp(upper_bound), expected_log_ratios)
                return jax.lax.cond(upper_bound_satisfies_constraint,
                                    lambda: (kl, jnp.exp(upper_bound), new_log_weights),
                                    lambda: (-1., -1., gmm_wrapper_state.gmm_state.log_weights))

            return jax.lax.cond(lower_bound == upper_bound,
                                lambda: (kl, eta, new_log_weights),
                                get_return_values_fn)

        def _update_weights(gmm_wrapper_state, expected_log_ratios, kl_bound):
            _, _, new_log_weights = _bracketing_search(gmm_wrapper_state,
                                                       expected_log_ratios,
                                                       kl_bound,
                                                       )
            return gmm_wrapper.replace_weights(gmm_wrapper_state, new_log_weights)

        gmm_wrapper_state2 = lax.cond(gmm_wrapper_state.gmm_state.num_components > 1,
                                      _update_weights,
                                      lambda gmm_wrapper_state, expected_log_ratios, kl_bound: gmm_wrapper_state,
                                      gmm_wrapper_state, expected_log_ratios, kl_bound)

        return gmm_wrapper_state2

    get_expected_log_ratios_fn = setup_get_expected_log_ratios(gmm_wrapper.log_densities_also_individual,
                                                               gmm_wrapper.store_rewards, TEMPERATURE, USE_SELF_NORMALIZED_IMPORTANCE_WEIGHTS)

    return WeightUpdater(init_weight_updater_state=init_trust_region_based_weight_updater,
                         update_weights=setup_update_weights_fn(get_expected_log_ratios_fn,
                                                                _update_weights_from_expected_log_ratios))
