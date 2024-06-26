from functools import partial
from typing import Tuple, NamedTuple, Callable, Optional
import chex
import jax.numpy as jnp
from algorithms.gmmvi.gmm_vi_utils.utils import reduce_weighted_logsumexp
from algorithms.gmmvi.optimization.least_squares import QuadRegression, QuadRegressionState
from algorithms.gmmvi.models.gmm_wrapper import GMMWrapperState, GMMWrapper
import jax
from jax._src.tree_util import Partial


class NgEstimatorState(NamedTuple):
    MORE_quad_reg_state: Optional[QuadRegressionState] = None


class NgEstimator(NamedTuple):
    init_ng_estimator_state: Callable
    get_expected_hessian_and_grad: Callable


def _get_rewards_for_comp(index: int, samples: chex.Array, mapping: chex.Array,
                          component_log_densities, log_ratios: chex.Array, log_ratio_grads: chex.Array,
                          background_densities: chex.Array) \
        -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
    # if ONLY_USE_OWN_SAMPLES:
    #     own_sample_indices = jnp.reshape(jnp.where(mapping == index), [-1])
    #     my_samples = samples[own_sample_indices]
    #     my_rewards = log_ratios[own_sample_indices]
    #     my_reward_grads = log_ratio_grads[own_sample_indices]
    #     my_background_densities = component_log_densities[index][own_sample_indices]
    #     my_component_log_densities = component_log_densities[index][own_sample_indices]
    #     return my_samples, my_rewards, my_reward_grads, my_background_densities, my_component_log_densities
    # else:
    return samples, log_ratios, log_ratio_grads, background_densities, component_log_densities[index]


def setup_stein_ng_estimator(gmm_wrapper: GMMWrapper, DIM, DIAGONAL_COVS,
                             ONLY_USE_OWN_SAMPLES: bool,
                             USE_SELF_NORMALIZED_IMPORTANCE_WEIGHTS):
    def init_stein_ng_estimator_state():
        return NgEstimatorState()

    def _stable_expectation(log_weights, log_values):
        n = jnp.array(jnp.shape(log_weights)[0], jnp.float32)
        lswe, signs = reduce_weighted_logsumexp(jnp.expand_dims(log_weights, 1) + jnp.log(jnp.abs(log_values)),
                                                w=jnp.sign(log_values), axis=0, return_sign=True)
        return 1 / n * signs * jnp.exp(lswe)

    def _get_expected_gradient_and_hessian_standard_iw(chol_cov, mean, component_log_densities, samples,
                                                       background_mixture_densities, log_ratio_grads):
        log_importance_weights = component_log_densities - background_mixture_densities
        expected_gradient = _stable_expectation(log_importance_weights, log_ratio_grads)

        if DIAGONAL_COVS:
            prec_times_diff = jnp.expand_dims(1 / (chol_cov ** 2), 1) \
                              * jnp.transpose(samples - mean)
            prec_times_diff_times_grad = jnp.transpose(prec_times_diff) * log_ratio_grads
        else:
            prec_times_diff = jax.scipy.linalg.cho_solve(chol_cov, jnp.transpose(samples - mean))
            prec_times_diff_times_grad = \
                jnp.expand_dims(jnp.transpose(prec_times_diff), 1) * jnp.expand_dims(log_ratio_grads, -1)
            log_importance_weights = jnp.expand_dims(log_importance_weights, 1)
        expected_hessian = _stable_expectation(log_importance_weights, prec_times_diff_times_grad)
        return expected_gradient, expected_hessian

    def _get_expected_gradient_and_hessian_self_normalized_iw(chol_cov, mean,
                                                              component_log_densities, samples,
                                                              background_mixture_densities, log_ratio_grads):
        log_weights = component_log_densities - background_mixture_densities
        log_weights -= jax.nn.logsumexp(log_weights, axis=0, keepdims=True)
        weights = jnp.exp(log_weights)
        importance_weights = weights / jnp.sum(weights, axis=0, keepdims=True)
        weighted_gradients = jnp.expand_dims(importance_weights, 1) * log_ratio_grads
        if DIAGONAL_COVS:
            prec_times_diff = jnp.expand_dims(1 / (chol_cov ** 2), 1) * jnp.transpose(samples - mean)
            expected_hessian = jnp.sum(jnp.transpose(prec_times_diff) * weighted_gradients, 0)
        else:
            prec_times_diff = jax.scipy.linalg.cho_solve((chol_cov, True), jnp.transpose(samples - mean))
            expected_hessian = jnp.sum(
                jnp.expand_dims(jnp.transpose(prec_times_diff), 1) * jnp.expand_dims(weighted_gradients, -1), 0)
            expected_hessian = 0.5 * (expected_hessian + jnp.transpose(expected_hessian))
        expected_gradient = jnp.sum(weighted_gradients, 0)
        return expected_gradient, expected_hessian

    def _get_expected_gradient_and_hessian_for_comp(gmm_wrapper_state: GMMWrapperState, i, my_component_log_densities,
                                                    my_samples, my_background_densities, my_log_ratios_grad):
        if USE_SELF_NORMALIZED_IMPORTANCE_WEIGHTS:
            expected_gradient, expected_hessian = \
                _get_expected_gradient_and_hessian_self_normalized_iw(gmm_wrapper_state.gmm_state.chol_covs[i], gmm_wrapper_state.gmm_state.means[i],
                                                                      my_component_log_densities, my_samples, my_background_densities, my_log_ratios_grad)
        else:
            expected_gradient, expected_hessian = \
                _get_expected_gradient_and_hessian_standard_iw(gmm_wrapper_state.gmm_state.chol_covs[i], gmm_wrapper_state.gmm_state.means[i],
                                                               my_component_log_densities, my_samples, my_background_densities, my_log_ratios_grad)
        return expected_gradient, expected_hessian

    @partial(jax.jit, static_argnums=(6,))
    def get_expected_hessian_and_grad(gmm_wrapper_state: GMMWrapperState,
                                      samples: chex.Array, mapping: chex.Array, background_densities: chex.Array,
                                      target_lnpdfs: chex.Array, target_lnpdfs_grads: chex.Array,
                                      num_components):

        relative_mapping = mapping - jnp.max(mapping) + num_components - 1

        model_densities, model_densities_grad, component_log_densities = jax.vmap(Partial(gmm_wrapper.log_density_and_grad, gmm_wrapper_state.gmm_state))(samples)
        component_log_densities = jnp.transpose(component_log_densities)
        log_ratios = target_lnpdfs - model_densities
        log_ratio_grads = target_lnpdfs_grads - model_densities_grad

        def _get_hessian_and_grad_per_comp(i, samples, relative_mapping, component_log_densities,
                                           log_ratios, log_ratio_grads, background_densities):
            my_samples, my_log_ratios, my_log_ratios_grad, my_background_densities, my_component_log_densities = \
                _get_rewards_for_comp(i, samples, relative_mapping, component_log_densities,
                                      log_ratios, log_ratio_grads, background_densities)

            return _get_expected_gradient_and_hessian_for_comp(gmm_wrapper_state, i,
                                                               my_component_log_densities,
                                                               my_samples,
                                                               my_background_densities,
                                                               my_log_ratios_grad)
        expected_gradient, expected_hessian = jax.vmap(_get_hessian_and_grad_per_comp, in_axes=(0, None, None, None, None, None, None))(jnp.arange(num_components),
                                                                                                                                        samples, relative_mapping, component_log_densities,
                                                                                                                                        log_ratios, log_ratio_grads, background_densities)
        return -expected_hessian, -expected_gradient

    return NgEstimator(init_ng_estimator_state=init_stein_ng_estimator_state,
                       get_expected_hessian_and_grad=get_expected_hessian_and_grad)


def setup_more_ng_estimator(gmm_wrapper: GMMWrapper, quad_regression: QuadRegression, DIM: int,
                            ONLY_USE_OWN_SAMPLES: bool, USE_SELF_NORMALIZED_IMPORTANCE_WEIGHTS: bool):
    def init_more_ng_estimator_state(quad_reg_state: QuadRegressionState):
        return NgEstimatorState(MORE_quad_reg_state=quad_reg_state)

    def get_expected_hessian_and_grad(gmm_wrapper_state: GMMWrapperState,
                                      samples: chex.Array, mapping: chex.Array,
                                      background_densities: chex.Array, target_lnpdfs: chex.Array,
                                      ) -> [chex.Array, chex.Array]:

        num_components = jnp.shape(gmm_wrapper_state.gmm_state.means)[0]

        expected_hessian_neg = jnp.empty((0,), dtype=jnp.float32)
        expected_gradient_neg = jnp.empty((0,), dtype=jnp.float32)
        relative_mapping = mapping - jnp.max(mapping) + num_components - 1

        model_densities, component_log_densities = gmm_wrapper.log_densities_also_individual(samples)

        log_ratios = target_lnpdfs - model_densities
        log_ratio_grads = jnp.zeros(jnp.shape(samples[0]))

        for i in jnp.arange(num_components):
            my_samples, my_rewards, _, my_background_densities, my_component_log_densities = \
                _get_rewards_for_comp(i, samples, relative_mapping, component_log_densities,
                                      log_ratios, log_ratio_grads, background_densities)

            log_weights = my_component_log_densities - my_background_densities
            if USE_SELF_NORMALIZED_IMPORTANCE_WEIGHTS:
                log_weights -= jax.nn.logsumexp(log_weights, axis=0, keepdims=True)
                weights = jnp.exp(log_weights)
                my_importance_weights = weights / jnp.sum(weights, axis=0, keepdims=True)
            else:
                my_importance_weights = jnp.exp(log_weights)
            reward_quad, reward_lin, const_term = quad_regression.fit_quadratic(gmm_wrapper_state.l2_regularizers[i],
                                                                                jnp.shape(my_samples)[0],
                                                                                my_samples, my_rewards,
                                                                                my_importance_weights,
                                                                                gmm_wrapper_state.gmm_state.means[i],
                                                                                gmm_wrapper_state.gmm_state.chol_covs[i])

            this_G_hat = reward_quad
            expected_hessian_neg = expected_hessian_neg.write(i, this_G_hat)
            this_g_hat = jnp.reshape(reward_quad @ jnp.expand_dims(gmm_wrapper_state.gmm_state.means[i], axis=1)
                                     - jnp.expand_dims(reward_lin, axis=1), [DIM])
            expected_gradient_neg = expected_gradient_neg.write(i, this_g_hat)
        expected_hessian_neg = expected_hessian_neg.stack()
        expected_gradient_neg = expected_gradient_neg.stack()
        return expected_hessian_neg, expected_gradient_neg

    return NgEstimator(init_ng_estimator_state=init_more_ng_estimator_state,
                       get_expected_hessian_and_grad=get_expected_hessian_and_grad)
