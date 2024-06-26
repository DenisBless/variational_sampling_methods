from typing import Union, NamedTuple, Callable, Optional
import chex
import jax.numpy as jnp
import numpyro.distributions as dist
import jax
from algorithms.gmmvi.models.gmm_wrapper import GMMWrapperState, GMMWrapper
from algorithms.gmmvi.optimization.sample_db import SampleDBState, SampleDB


class ComponentAdaptationState(NamedTuple):
    num_calls_to_add_heuristic: Optional[int]
    reward_improvements: Optional[chex.Array]


class ComponentAdaptation(NamedTuple):
    init_component_adaptation: Callable
    adapt_number_of_components: Callable


def setup_fixed_component_adaptation():
    def init_fixed_component_adaptation_state():
        return ComponentAdaptationState(num_calls_to_add_heuristic=None,
                                        reward_improvements=None,
                                        )

    def adapt_number_of_components_fixed(component_adaption_state: ComponentAdaptationState,
                                         sample_db_state: SampleDBState,
                                         gmm_wrapper_state: GMMWrapperState,
                                         iteration, seed):
        return gmm_wrapper_state, component_adaption_state

    return ComponentAdaptation(init_component_adaptation=init_fixed_component_adaptation_state,
                               adapt_number_of_components=adapt_number_of_components_fixed)


def setup_vips_component_adaptation(sample_db: SampleDB, gmm_wrapper: GMMWrapper, target_log_prob,
                                    DIM: int, PRIOR_MEAN: Union[float, chex.Array],
                                    PRIOR_COV: Union[float, chex.Array], DIAGONAL_COVS: bool,
                                    DEL_ITERS: int, ADD_ITERS: int,
                                    MAX_COMPONENTS: int, THRESHOLD_FOR_ADD_HEURISTIC: chex.Array,
                                    MIN_WEIGHT_FOR_DEL_HEURISTIC: float, NUM_DATABASE_SAMPLES: int,
                                    NUM_PRIOR_SAMPLES: int):
    THRESHOLD_FOR_ADD_HEURISTIC = jnp.array(THRESHOLD_FOR_ADD_HEURISTIC)

    if (PRIOR_MEAN is not None) and (PRIOR_COV is not None):
        if jnp.ndim(PRIOR_COV) == 0:
            PRIOR_CHOL_COV = jnp.sqrt(PRIOR_COV * jnp.ones(DIM))
        if jnp.ndim(PRIOR_MEAN) == 0:
            PRIOR_MEAN = PRIOR_MEAN * jnp.ones(DIM)
    else:
        PRIOR_CHOL_COV = None
        PRIOR_MEAN = None

    FILTER_DELAY = jnp.array(jnp.floor(DEL_ITERS / 3), dtype=jnp.int32)
    GAUSSIAN = dist.Normal(0, jnp.array(DEL_ITERS / 8., jnp.float32))
    KERNEL = jnp.exp(GAUSSIAN.log_prob(jnp.arange(start=-FILTER_DELAY, stop=FILTER_DELAY, dtype=jnp.float32)))
    KERNEL = jnp.reshape(KERNEL / jnp.sum(KERNEL), [-1, 1, 1])

    def init_vips_component_adaption_state():
        return ComponentAdaptationState(num_calls_to_add_heuristic=0,
                                        reward_improvements=jnp.array(jnp.zeros(0), dtype=jnp.float32))

    def _sample_from_prior(num_samples: int, seed):
        return jnp.transpose(
            jnp.expand_dims(PRIOR_MEAN, axis=-1) + jnp.expand_dims(PRIOR_CHOL_COV, -1) * jax.random.normal(seed, (
            DIM, num_samples)))

    def _select_samples_for_adding_heuristic(component_adaption_state: ComponentAdaptationState,
                                             sample_db_state: SampleDBState, seed):

        key, subkey = jax.random.split(seed)
        samples, target_lnpdfs = sample_db.get_random_sample(sample_db_state, NUM_DATABASE_SAMPLES, subkey)
        prior_samples = jnp.zeros((0, DIM), jnp.float32)
        key, subkey = jax.random.split(key)
        if NUM_PRIOR_SAMPLES > 0:
            prior_samples = _sample_from_prior(NUM_PRIOR_SAMPLES, subkey)
            sample_db_state = sample_db.update_num_samples_written(sample_db_state, (
                        sample_db_state.num_samples_written + NUM_PRIOR_SAMPLES))
        component_adaption_state = ComponentAdaptationState(
            num_calls_to_add_heuristic=component_adaption_state.num_calls_to_add_heuristic + 1,
            reward_improvements=component_adaption_state.reward_improvements,
        )

        return component_adaption_state, sample_db_state, samples, target_lnpdfs, prior_samples

    def _add_at_best_location(component_adaption_state: ComponentAdaptationState, gmm_wrapper_state: GMMWrapperState,
                              samples, target_lnpdfs, seed: chex.PRNGKey):
        def _get_prior_entropy(chol_cov):
            avg_entropy = 0.5 * DIM * (jnp.log(2 * jnp.pi) + 1) + jnp.sum(jnp.log(chol_cov))
            return avg_entropy

        it = component_adaption_state.num_calls_to_add_heuristic % len(THRESHOLD_FOR_ADD_HEURISTIC)
        model_log_densities = jax.vmap(gmm_wrapper.log_density, in_axes=(None, 0))(gmm_wrapper_state.gmm_state, samples)
        init_weight = 1e-29
        a = jax.random.uniform(key=seed, shape=(1,), )
        if PRIOR_MEAN is not None and PRIOR_CHOL_COV is not None:
            des_entropy = gmm_wrapper.average_entropy(gmm_wrapper_state.gmm_state) * a + _get_prior_entropy(PRIOR_CHOL_COV) * (1 - a)
        else:
            des_entropy = gmm_wrapper.average_entropy(gmm_wrapper_state.gmm_state)
        max_logdensity = jnp.max(model_log_densities)
        rewards = target_lnpdfs - jnp.maximum(max_logdensity - THRESHOLD_FOR_ADD_HEURISTIC[it], model_log_densities)
        new_mean = samples[jnp.argmax(rewards)]
        H_unscaled = 0.5 * DIM * (jnp.log(2.0 * jnp.pi) + 1)
        c = jnp.exp((2 * (des_entropy - H_unscaled)) / DIM)
        if DIAGONAL_COVS:
            new_cov = c * jnp.ones(DIM)
        else:
            new_cov = c * jnp.eye(DIM)

        return gmm_wrapper.add_component(gmm_wrapper_state, init_weight, new_mean, new_cov,
                                         jnp.reshape(THRESHOLD_FOR_ADD_HEURISTIC[it], [1]),
                                         )

    @jax.jit
    def _add_new_component(component_adaption_state: ComponentAdaptationState, sample_db_state: SampleDBState,
                           gmm_wrapper_state: GMMWrapperState, key):
        key, subkey = jax.random.split(key)
        component_adaption_state, sample_db_state, samples, target_lnpdfs, prior_samples = _select_samples_for_adding_heuristic(
            component_adaption_state, sample_db_state, subkey)
        if NUM_PRIOR_SAMPLES > 0:
            samples = jnp.concatenate((samples, prior_samples), 0)
            target_lnpdfs = jnp.concatenate((target_lnpdfs, target_log_prob(prior_samples)), 0)

        key, subkey = jax.random.split(key)
        return _add_at_best_location(component_adaption_state, gmm_wrapper_state, samples, target_lnpdfs,
                                     subkey), component_adaption_state, sample_db_state

    def _delete_bad_components(components_adaption_state: ComponentAdaptationState,
                               gmm_wrapper_state: GMMWrapperState):
        # estimate the relative improvement for every component with respect to
        # the improvement it would need to catch up (assuming linear improvement) with the best component
        current_smoothed_reward = jnp.mean(gmm_wrapper_state.reward_history[:, -jnp.size(KERNEL):] * jnp.reshape(KERNEL, [1, -1]), axis=1)
        old_smoothed_reward = jnp.mean(gmm_wrapper_state.reward_history[:, -jnp.size(KERNEL) - DEL_ITERS:-DEL_ITERS] * jnp.reshape(KERNEL, [1, -1]), axis=1)

        old_smoothed_reward -= jnp.max(current_smoothed_reward)
        current_smoothed_reward -= jnp.max(current_smoothed_reward)
        reward_improvements = (current_smoothed_reward - old_smoothed_reward) / jnp.abs(old_smoothed_reward)

        # compute for each component the maximum weight it had within the last del_iters,
        # or that it would have gotten when we used greedy updates
        max_actual_weights = jnp.max(gmm_wrapper_state.weight_history[:, -jnp.size(KERNEL) - DEL_ITERS:-1], axis=1)
        max_greedy_weights = jnp.max(jnp.exp(gmm_wrapper_state.reward_history[:, -jnp.size(KERNEL) - DEL_ITERS:]
                                             - jax.nn.logsumexp(gmm_wrapper_state.reward_history[:, -jnp.size(KERNEL) - DEL_ITERS:], axis=0, keepdims=True)), axis=1)
        max_weights = jnp.maximum(max_actual_weights, max_greedy_weights)

        is_stagnating = reward_improvements <= 0.4
        is_low_weight = max_weights < MIN_WEIGHT_FOR_DEL_HEURISTIC
        is_old_enough = gmm_wrapper_state.reward_history[:, -DEL_ITERS] != -jnp.finfo(jnp.float32).max
        is_bad = jnp.all(jnp.vstack((is_stagnating, is_low_weight, is_old_enough)), axis=0)
        bad_component_indices = jnp.where(is_bad)[0]

        for idx in jnp.sort(bad_component_indices, descending=True):
            gmm_wrapper_state = gmm_wrapper.remove_component(gmm_wrapper_state, idx)

        components_adaption_state = ComponentAdaptationState(
            reward_improvements=reward_improvements,
            num_calls_to_add_heuristic=components_adaption_state.num_calls_to_add_heuristic,
        )

        return gmm_wrapper_state, components_adaption_state

    def adapt_number_of_components(component_adaption_state: ComponentAdaptationState,
                                   sample_db_state: SampleDBState,
                                   gmm_wrapper_state: GMMWrapperState,
                                   iteration, seed):
        if iteration > DEL_ITERS:
            gmm_wrapper_state, component_adaption_state = _delete_bad_components(component_adaption_state, gmm_wrapper_state)
        if iteration > 1 and iteration % ADD_ITERS == 0:
            if gmm_wrapper_state.gmm_state.num_components < MAX_COMPONENTS:
                gmm_wrapper_state, component_adaption_state, sample_db_state = _add_new_component(component_adaption_state, sample_db_state,
                                                                                 gmm_wrapper_state, seed)

        return gmm_wrapper_state, component_adaption_state, sample_db_state

    return ComponentAdaptation(init_component_adaptation=init_vips_component_adaption_state,
                               adapt_number_of_components=adapt_number_of_components)
