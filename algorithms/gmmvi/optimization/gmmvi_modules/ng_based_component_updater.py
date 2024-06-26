from typing import NamedTuple, Callable
import chex
import jax.numpy as jnp
from algorithms.gmmvi.models.gmm_wrapper import GMMWrapperState, GMMWrapper
import jax


class NgBasedComponentUpdaterState(NamedTuple):
    pass


class NgBasedComponentUpdater(NamedTuple):
    init_ng_based_component_updater_state: Callable
    apply_NG_update: Callable


def setup_direct_ng_based_component_updater(gmm_wrapper: GMMWrapper, DIM, DIAGONAL_COVS, TEMPERATURE: float,
                                            INITIAL_REGULARIZER):
    def init_direct_ng_based_component_updater_state():
        return NgBasedComponentUpdaterState()

    def apply_ng_update(gmm_wrapper_state: GMMWrapperState, expected_hessians_neg: chex.Array,
                        expected_gradients_neg: chex.Array, stepsizes: chex.Array):
        means = jnp.empty((gmm_wrapper_state.gmm_state.num_components,), dtype=jnp.float32)
        chols = jnp.empty((gmm_wrapper_state.gmm_state.num_components,), dtype=jnp.float32)
        successes = jnp.full((gmm_wrapper_state.gmm_state.num_components,), False)
        for i in range(gmm_wrapper_state.gmm_state.num_components):
            old_chol = gmm_wrapper_state.gmm_state.chol_covs[i]
            old_mean = gmm_wrapper_state.gmm_state.means[i]
            old_inv_chol = jnp.linalg.inv(old_chol)
            old_precision = jnp.transpose(old_inv_chol) @ old_inv_chol
            old_lin = old_precision @ jnp.expand_dims(old_mean, 1)

            delta_precision = expected_hessians_neg[i]
            delta_lin = expected_hessians_neg[i] @ jnp.expand_dims(old_mean, 1) - jnp.expand_dims(expected_gradients_neg[i], 1)

            new_lin = old_lin + stepsizes[i] * delta_lin
            new_precision = old_precision + stepsizes[i] * delta_precision
            new_mean = jnp.reshape(jnp.linalg.solve(new_precision, new_lin), [-1])
            new_cov = jnp.linalg.inv(new_precision)
            new_chol = jnp.linalg.cholesky(new_cov)

            if jnp.any(jnp.isnan(new_chol)):
                success = False
                new_mean = old_mean
                new_chol = old_chol
            else:
                success = True

            means = means.at[i].set(new_mean)
            chols = chols.at[i].set(new_chol)
            successes = successes.at[i].set(success)

        chols = jnp.stack(chols)
        means = jnp.stack(means)
        successes = jnp.stack(successes)

        updated_l2_reg = jnp.where(successes,
                                   jnp.maximum(0.5 * gmm_wrapper_state.l2_regularizers, INITIAL_REGULARIZER),
                                   jnp.minimum(1e-6, 10 * gmm_wrapper_state.l2_regularizers))

        new_gmm_state = gmm_wrapper.replace_components(gmm_wrapper_state.gmm_state, means, chols)

        return GMMWrapperState(adding_thresholds=gmm_wrapper_state.adding_thresholds,
                               gmm_state=new_gmm_state,
                               l2_regularizers=updated_l2_reg,
                               num_received_updates=gmm_wrapper_state.num_received_updates + gmm_wrapper_state.gmm_state.num_components,
                               last_log_etas=gmm_wrapper_state.last_log_etas,
                               max_component_id=gmm_wrapper_state.max_component_id,
                               reward_history=gmm_wrapper_state.reward_history,
                               stepsizes=gmm_wrapper_state.stepsizes,
                               unique_component_ids=gmm_wrapper_state.unique_component_ids,
                               weight_history=gmm_wrapper_state.weight_history,
                               )

    return NgBasedComponentUpdater(init_ng_based_component_updater_state=init_direct_ng_based_component_updater_state,
                                   apply_NG_update=apply_ng_update)


def setup_ng_based_component_updater_iblr(gmm_wrapper: GMMWrapper, DIM, DIAGONAL_COVS, TEMPERATURE: float,
                                          INITIAL_REGULARIZER):
    def init_ng_based_component_updater_iblr_state():
        return NgBasedComponentUpdaterState()

    def apply_ng_update(gmm_wrapper_state: GMMWrapperState, expected_hessians_neg: chex.Array,
                        expected_gradients_neg: chex.Array, stepsizes: chex.Array):
        means = jnp.empty((gmm_wrapper_state.gmm_state.num_components,), dtype=jnp.float32)
        chols = jnp.empty((gmm_wrapper_state.gmm_state.num_components,), dtype=jnp.float32)
        successes = jnp.full((gmm_wrapper_state.gmm_state.num_components,), False)

        for i in range(gmm_wrapper_state.gmm_state.num_components):
            old_chol = gmm_wrapper_state.gmm_state.chol_covs[i]
            old_mean = gmm_wrapper_state.gmm_state.means[i]

            if DIAGONAL_COVS:
                correction_term = stepsizes[i] / 2 * expected_hessians_neg[i] * old_chol * old_chol * expected_hessians_neg[i]
                old_inv_chol = 1. / old_chol
                old_precision = old_inv_chol * old_inv_chol
            else:
                correction_term = stepsizes[i] / 2 * expected_hessians_neg[i] @ old_chol @ jnp.transpose(old_chol) @ expected_hessians_neg[i]
                old_inv_chol = jnp.linalg.inv(old_chol)
                old_precision = jnp.transpose(old_inv_chol) @ old_inv_chol

            delta_precision = expected_hessians_neg[i] + correction_term
            delta_mean = - expected_gradients_neg[i]

            if gmm_wrapper_state.num_received_updates[i] == 0:
                # first time: no mean update
                new_mean = old_mean
            else:
                if DIAGONAL_COVS:
                    new_mean = old_mean + stepsizes[i] * old_chol * old_chol * delta_mean
                else:
                    new_mean = old_mean + jnp.squeeze(
                        stepsizes[i] * old_chol @ jnp.transpose(old_chol) @ jnp.expand_dims(delta_mean, 1))

            new_precision = old_precision + stepsizes[i] * delta_precision
            if DIAGONAL_COVS:
                new_cov = 1. / new_precision
                new_chol = jnp.sqrt(new_cov)
            else:
                new_cov = jnp.linalg.inv(new_precision)
                new_chol = jnp.linalg.cholesky(new_cov)

            if jnp.any(jnp.isnan(new_chol)):
                success = False
                new_mean = old_mean
                new_chol = old_chol
            else:
                success = True

            means = means.at[i].set(new_mean)
            chols = chols.at[i].set(new_chol)
            successes = chols.at[i].set(success)

        chols = jnp.stack(chols)
        means = jnp.stack(chols)
        successes = jnp.stack(successes)

        updated_l2_reg = jnp.where(successes,
                                   jnp.maximum(0.5 * gmm_wrapper_state.l2_regularizers, INITIAL_REGULARIZER),
                                   jnp.minimum(1e-6, 10 * gmm_wrapper_state.l2_regularizers))

        new_gmm_state = gmm_wrapper.replace_components(gmm_wrapper_state, means, chols)

        return GMMWrapperState(weight_history=gmm_wrapper_state.weight_history,
                               adding_thresholds=gmm_wrapper_state.adding_thresholds,
                               gmm_state=new_gmm_state,
                               l2_regularizers=updated_l2_reg,
                               num_received_updates=gmm_wrapper_state.num_received_updates + gmm_wrapper_state.gmm_state.num_components,
                               last_log_etas=gmm_wrapper_state.last_log_etas,
                               max_component_id=gmm_wrapper_state.max_component_id,
                               reward_history=gmm_wrapper_state.reward_history,
                               stepsizes=gmm_wrapper_state.stepsizes,
                               unique_component_ids=gmm_wrapper_state.unique_component_ids,)

    return NgBasedComponentUpdater(init_ng_based_component_updater_state=init_ng_based_component_updater_iblr_state,
                                   apply_NG_update=apply_ng_update)


def setup_kl_constrained_ng_based_component_updater(gmm_wrapper: GMMWrapper, DIM, DIAGONAL_COVS, TEMPERATURE: float,
                                                    INITIAL_REGULARIZER):
    def init_kl_constrained_ng_based_component_updater():
        return NgBasedComponentUpdaterState()

    def _kl(eta: jnp.float32, old_lin_term: chex.Array, old_precision: chex.Array, old_inv_chol: chex.Array,
            reward_lin: chex.Array, reward_quad: chex.Array, kl_const_part: jnp.float32, old_mean: chex.Array,
            eta_in_logspace: bool) -> [jnp.float32, chex.Array, chex.Array, chex.Array]:

        eta = jax.lax.cond(eta_in_logspace,
                           lambda eta: jnp.exp(eta),
                           lambda eta: eta,
                           eta)

        new_lin = (eta * old_lin_term + reward_lin) / eta
        new_precision = (eta * old_precision + reward_quad) / eta
        if DIAGONAL_COVS:
            chol_precision = jnp.sqrt(new_precision)
            new_mean = 1./new_precision * new_lin
            inv_chol_inv = 1./chol_precision
            diff = old_mean - new_mean
            # this is numerically more stable:
            kl = 0.5 * (jnp.maximum(0., jnp.sum(jnp.log(new_precision / old_precision)
                        + old_precision / new_precision) - DIM)
                        + jnp.sum(jnp.square(old_inv_chol * diff)))
        else:
            chol_precision = jnp.linalg.cholesky(new_precision)

            def true_fn():
                new_mean = old_mean
                inv_chol_inv = old_inv_chol
                new_precision = old_precision
                kl = jnp.finfo(jnp.float32).max

                return kl, new_mean, new_precision, inv_chol_inv

            def false_fn():
                new_mean = jnp.reshape(jax.scipy.linalg.cho_solve((chol_precision, True), jnp.expand_dims(new_lin, 1)),
                                       [-1])
                inv_chol_inv = jnp.linalg.inv(chol_precision)

                new_logdet = -2 * jnp.sum(jnp.log(jnp.diag(chol_precision)))
                trace_term = jnp.square(jnp.linalg.norm(inv_chol_inv @ jnp.transpose(old_inv_chol)))
                diff = old_mean - new_mean
                kl = 0.5 * (kl_const_part - new_logdet + trace_term + jnp.sum(jnp.square(jnp.dot(old_inv_chol, diff))))

                return kl, new_mean, new_precision, inv_chol_inv

            kl, new_mean, new_precision, inv_chol_inv = jax.lax.cond(jnp.any(jnp.isnan(chol_precision)),
                                                                     true_fn,
                                                                     false_fn)

        return kl, new_mean, new_precision, inv_chol_inv

    # always in log_space
    def _bracketing_search(KL_BOUND: jnp.float32, lower_bound: jnp.float32,
                           upper_bound: jnp.float32, old_lin_term: chex.Array, old_precision: chex.Array,
                           old_inv_chol: chex.Array, reward_lin_term: chex.Array, reward_quad_term: chex.Array,
                           kl_const_part: jnp.float32, old_mean: chex.Array) -> [jnp.float32, jnp.float32]:

        def cond_fn(carry):
            it, lower_bound, upper_bound, eta, kl, _ = carry
            diff = jnp.minimum(jnp.exp(upper_bound) - jnp.exp(eta), jnp.exp(eta) - jnp.exp(lower_bound))
            return (it < 1000) & (diff >= 1e-1) & ((jnp.abs(KL_BOUND - kl) >= 1e-1 * KL_BOUND) | jnp.isnan(kl))

        def body_fn(carry):
            it, lower_bound, upper_bound, eta, _, upper_bound_satisfies_constraint = carry
            kl = _kl(eta, old_lin_term, old_precision, old_inv_chol, reward_lin_term,
                     reward_quad_term, kl_const_part, old_mean, True)[0]

            def true_fn():
                new_lower_bound = new_upper_bound = eta
                return it+1, new_lower_bound, new_upper_bound, eta, kl, upper_bound_satisfies_constraint

            def false_fn():
                new_upper_bound, new_lower_bound, new_upper_bound_satisfies_constraint = jax.lax.cond(KL_BOUND > kl,
                                                                                                      lambda upper_bound, lower_bound, eta: (eta, lower_bound, True),
                                                                                                      lambda upper_bound, lower_bound, eta: (upper_bound, eta, False),
                                                                                                      upper_bound,
                                                                                                      lower_bound,
                                                                                                      eta)
                new_eta = 0.5 * (new_upper_bound + new_lower_bound)
                return it+1, new_lower_bound, new_upper_bound, new_eta, kl, new_upper_bound_satisfies_constraint

            return jax.lax.cond(jnp.abs(KL_BOUND - kl) < 1e-1 * KL_BOUND, true_fn, false_fn)

        _, lower_bound, upper_bound, eta, kl, upper_bound_satisfies_constraint = jax.lax.while_loop(cond_fn, body_fn, init_val=(0, lower_bound, upper_bound, 0.5 * (upper_bound + lower_bound), -1000, False))

        lower_bound = jax.lax.cond(upper_bound_satisfies_constraint,
                                   lambda lower_bound, upper_bound: upper_bound,
                                   lambda lower_bound, upper_bound: lower_bound,
                                   lower_bound,
                                   upper_bound)

        return jnp.exp(lower_bound), jnp.exp(upper_bound)

    @jax.jit
    def apply_ng_update(gmm_wrapper_state: GMMWrapperState, expected_hessians_neg: chex.Array,
                        expected_gradients_neg: chex.Array, stepsizes: chex.Array):

        def _apply_gn_update_per_comp(old_chol, old_mean, last_eta, eps, reward_quad, expected_gradients_neg):
            if DIAGONAL_COVS:
                reward_lin = reward_quad * old_mean - expected_gradients_neg
                old_logdet = 2 * jnp.sum(jnp.log(old_chol))
                old_inv_chol = 1./old_chol
                old_precision = old_inv_chol**2
                old_lin_term = old_precision * old_mean
                kl_const_part = old_logdet - DIM
            else:
                reward_lin = jnp.squeeze(reward_quad @ jnp.expand_dims(old_mean, 1)) - expected_gradients_neg
                old_logdet = 2 * jnp.sum(jnp.log(jnp.diag(old_chol)))
                old_inv_chol = jnp.linalg.inv(old_chol)
                old_precision = jnp.transpose(old_inv_chol) @ old_inv_chol
                old_lin_term = jnp.dot(old_precision, old_mean)
                kl_const_part = old_logdet - DIM

            lower_bound_const, upper_bound_const = jax.lax.cond(last_eta < 0,
                                                                lambda last_eta: (jnp.array(-20.), jnp.array(80.)),
                                                                lambda last_eta: (jnp.maximum(0., jnp.log(last_eta) - 3), jnp.log(last_eta) + 3),
                                                                last_eta)

            new_lower, new_upper = _bracketing_search(eps, lower_bound_const, upper_bound_const,
                                                      old_lin_term, old_precision, old_inv_chol, reward_lin,
                                                      reward_quad, kl_const_part, old_mean)
            eta = jnp.maximum(new_lower, TEMPERATURE)

            def true_lower_equals_upper():
                new_kl, new_mean, _, new_inv_chol_inv = _kl(eta, old_lin_term, old_precision,
                                                            old_inv_chol, reward_lin, reward_quad,
                                                            kl_const_part, old_mean, False)
                if DIAGONAL_COVS:
                    new_cov = jnp.square(new_inv_chol_inv)
                    new_chol = jnp.sqrt(new_cov)
                else:
                    new_cov = jnp.transpose(new_inv_chol_inv) @ new_inv_chol_inv
                    new_chol = jnp.linalg.cholesky(new_cov)

                return jax.lax.cond((new_kl < jnp.finfo(jnp.float32).max) & (~jnp.any(jnp.isnan(new_chol))),
                                    lambda: (True, new_mean, new_chol, new_kl),
                                    lambda: (False, old_mean, old_chol, -1.),   # values will be ignored anyway, if success is false
                                    )

            success, new_mean, new_chol, new_kl = jax.lax.cond(new_lower == new_upper,
                                                               true_lower_equals_upper,
                                                               lambda: (False, old_mean, old_chol, -1.))
            return jax.lax.cond(success,
                                lambda: (new_chol, new_mean, new_kl, True, eta),
                                lambda: (old_chol, old_mean, -1., False, -1.))

        chols, means, kls, successes, etas = jax.vmap(_apply_gn_update_per_comp)(gmm_wrapper_state.gmm_state.chol_covs,
                                                                                 gmm_wrapper_state.gmm_state.means,
                                                                                 gmm_wrapper_state.last_log_etas,
                                                                                 stepsizes,
                                                                                 expected_hessians_neg,
                                                                                 expected_gradients_neg)

        new_gmm_state = gmm_wrapper.replace_components(gmm_wrapper_state.gmm_state, means, chols)
        updated_l2_reg = jnp.where(successes,
                                   jnp.maximum(0.5 * gmm_wrapper_state.l2_regularizers, INITIAL_REGULARIZER),
                                   jnp.minimum(1e-6, 10 * gmm_wrapper_state.l2_regularizers))

        return GMMWrapperState(adding_thresholds=gmm_wrapper_state.adding_thresholds,
                               gmm_state=new_gmm_state,
                               l2_regularizers=updated_l2_reg,
                               last_log_etas=etas,
                               num_received_updates=gmm_wrapper_state.num_received_updates + 1,
                               max_component_id=gmm_wrapper_state.max_component_id,
                               reward_history=gmm_wrapper_state.reward_history,
                               stepsizes=gmm_wrapper_state.stepsizes,
                               unique_component_ids=gmm_wrapper_state.unique_component_ids,
                               weight_history=gmm_wrapper_state.weight_history,
                               )

    return NgBasedComponentUpdater(init_ng_based_component_updater_state=init_kl_constrained_ng_based_component_updater,
                                   apply_NG_update=apply_ng_update)
