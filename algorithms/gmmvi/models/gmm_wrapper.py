from typing import NamedTuple, Callable
import chex
from algorithms.gmmvi.models.gmm import GMMState, GMM
import jax.numpy as jnp


class GMMWrapperState(NamedTuple):
    gmm_state: GMMState
    l2_regularizers: chex.Array
    last_log_etas: chex.Array
    num_received_updates: chex.Array
    stepsizes: chex.ArrayTree
    reward_history: chex.Array
    weight_history: chex.Array
    unique_component_ids: chex.Array
    max_component_id: chex.Array
    adding_thresholds: chex.Array


class GMMWrapper(NamedTuple):
    init_gmm_wrapper_state: Callable
    add_component: Callable
    add_component: Callable
    remove_component: Callable
    replace_components: Callable
    store_rewards: Callable
    update_stepsizes: Callable
    replace_weights: Callable
    log_density: Callable
    average_entropy: Callable
    log_densities_also_individual: Callable
    component_log_densities: Callable
    sample_from_components_no_shuffle: Callable
    log_density_and_grad: Callable
    sample: Callable


def setup_gmm_wrapper(gmm: GMM, INITIAL_STEPSIZE, INITIAL_REGULARIZER, MAX_REWARD_HISTORY_LENGTH, INITIAL_LAST_ETA=-1):
    def init_gmm_wrapper_state(gmm_state: GMMState):
        return GMMWrapperState(gmm_state=gmm_state,
                               l2_regularizers=INITIAL_REGULARIZER * jnp.ones(gmm_state.num_components),
                               last_log_etas=INITIAL_LAST_ETA * jnp.ones(gmm_state.num_components),
                               num_received_updates=jnp.zeros(gmm_state.num_components),
                               stepsizes=INITIAL_STEPSIZE * jnp.ones(gmm_state.num_components),
                               reward_history=jnp.finfo(jnp.float32).min * jnp.ones(
                                   (gmm_state.num_components, MAX_REWARD_HISTORY_LENGTH)),
                               weight_history=jnp.finfo(jnp.float32).min * jnp.ones(
                                   (gmm_state.num_components, MAX_REWARD_HISTORY_LENGTH)),
                               unique_component_ids=jnp.arange(gmm_state.num_components),
                               max_component_id=jnp.max(jnp.arange(gmm_state.num_components)),
                               adding_thresholds=-jnp.ones(gmm_state.num_components))

    def add_component(gmm_wrapper_state: GMMWrapperState, initial_weight: jnp.float32, initial_mean: chex.Array,
                      initial_cov: chex.Array, adding_threshold: chex.Array):

        return GMMWrapperState(gmm_state=gmm.add_component(gmm_wrapper_state.gmm_state, initial_weight, initial_mean, initial_cov),
                               l2_regularizers=jnp.concatenate((gmm_wrapper_state.l2_regularizers, jnp.ones(1) * INITIAL_REGULARIZER), axis=0),
                               last_log_etas=jnp.concatenate((gmm_wrapper_state.last_log_etas, jnp.ones(1) * INITIAL_LAST_ETA), axis=0),
                               num_received_updates=jnp.concatenate((gmm_wrapper_state.num_received_updates, jnp.zeros(1)), axis=0),
                               stepsizes=jnp.concatenate((gmm_wrapper_state.stepsizes, jnp.ones(1) * INITIAL_STEPSIZE), axis=0),
                               reward_history=jnp.concatenate((gmm_wrapper_state.reward_history, jnp.ones((1, MAX_REWARD_HISTORY_LENGTH)) * jnp.finfo(jnp.float32).min), axis=0),
                               weight_history=jnp.concatenate((gmm_wrapper_state.weight_history, jnp.ones((1, MAX_REWARD_HISTORY_LENGTH)) * initial_weight), axis=0),
                               unique_component_ids=jnp.concatenate((gmm_wrapper_state.unique_component_ids, jnp.ones(1, dtype=jnp.int32) * gmm_wrapper_state.max_component_id), axis=0),
                               max_component_id=gmm_wrapper_state.max_component_id + 1,
                               adding_thresholds=jnp.concatenate((gmm_wrapper_state.adding_thresholds, adding_threshold), axis=0))

    def remove_component(gmm_wrapper_state: GMMWrapperState, idx: int):
        return GMMWrapperState(
            gmm_state=gmm.remove_component(gmm_wrapper_state.gmm_state, idx),
            max_component_id=gmm_wrapper_state.max_component_id,
            unique_component_ids=jnp.concatenate((gmm_wrapper_state.unique_component_ids[:idx], gmm_wrapper_state.unique_component_ids[idx + 1:]), axis=0),
            l2_regularizers=jnp.concatenate((gmm_wrapper_state.l2_regularizers[:idx], gmm_wrapper_state.l2_regularizers[idx + 1:]), axis=0),
            last_log_etas=jnp.concatenate((gmm_wrapper_state.last_log_etas[:idx], gmm_wrapper_state.last_log_etas[idx + 1:]), axis=0),
            num_received_updates=jnp.concatenate((gmm_wrapper_state.num_received_updates[:idx], gmm_wrapper_state.num_received_updates[idx + 1:]), axis=0),
            stepsizes=jnp.concatenate((gmm_wrapper_state.stepsizes[:idx], gmm_wrapper_state.stepsizes[idx + 1:]), axis=0),
            reward_history=jnp.concatenate((gmm_wrapper_state.reward_history[:idx], gmm_wrapper_state.reward_history[idx + 1:]), axis=0),
            weight_history=jnp.concatenate((gmm_wrapper_state.weight_history[:idx], gmm_wrapper_state.weight_history[idx + 1:]), axis=0),
            adding_thresholds=jnp.concatenate((gmm_wrapper_state.adding_thresholds[:idx], gmm_wrapper_state.adding_thresholds[idx + 1:]), axis=0),
            )

    def update_weights(gmm_wrapper_state: GMMWrapperState, new_log_weights: chex.Array):
        return GMMWrapperState(gmm_state=gmm.replace_weights(gmm_wrapper_state.gmm_state, new_log_weights),
                               weight_history=jnp.concatenate((gmm_wrapper_state.weight_history[:, 1:],
                                                               jnp.expand_dims(jnp.exp(gmm_wrapper_state.gmm_state.log_weights), 1)), axis=1),
                               l2_regularizers=gmm_wrapper_state.l2_regularizers,
                               last_log_etas=gmm_wrapper_state.last_log_etas,
                               num_received_updates=gmm_wrapper_state.num_received_updates,
                               stepsizes=gmm_wrapper_state.stepsizes,
                               reward_history=gmm_wrapper_state.reward_history,
                               unique_component_ids=gmm_wrapper_state.unique_component_ids,
                               max_component_id=gmm_wrapper_state.max_component_id,
                               adding_thresholds=gmm_wrapper_state.adding_thresholds)

    def update_rewards(gmm_wrapper_state: GMMWrapperState, rewards: chex.Array):
        return GMMWrapperState(gmm_state=gmm_wrapper_state.gmm_state,
                               l2_regularizers=gmm_wrapper_state.l2_regularizers,
                               last_log_etas=gmm_wrapper_state.last_log_etas,
                               num_received_updates=gmm_wrapper_state.num_received_updates,
                               stepsizes=gmm_wrapper_state.stepsizes,
                               reward_history=jnp.concatenate((gmm_wrapper_state.reward_history[:, 1:],
                                                               jnp.expand_dims(rewards, 1)), axis=1),
                               weight_history=gmm_wrapper_state.weight_history,
                               unique_component_ids=gmm_wrapper_state.unique_component_ids,
                               max_component_id=gmm_wrapper_state.max_component_id,
                               adding_thresholds=gmm_wrapper_state.adding_thresholds,
                               )

    def update_stepsizes(gmm_wrapper_state: GMMWrapperState, new_stepsizes: chex.Array):
        return GMMWrapperState(gmm_state=gmm_wrapper_state.gmm_state,
                               l2_regularizers=gmm_wrapper_state.l2_regularizers,
                               last_log_etas=gmm_wrapper_state.last_log_etas,
                               num_received_updates=gmm_wrapper_state.num_received_updates,
                               stepsizes=new_stepsizes,
                               reward_history=gmm_wrapper_state.reward_history,
                               weight_history=gmm_wrapper_state.weight_history,
                               unique_component_ids=gmm_wrapper_state.unique_component_ids,
                               max_component_id=gmm_wrapper_state.max_component_id,
                               adding_thresholds=gmm_wrapper_state.adding_thresholds,
                               )

    return GMMWrapper(init_gmm_wrapper_state=init_gmm_wrapper_state,
                      add_component=add_component,
                      remove_component=remove_component,
                      store_rewards=update_rewards,
                      update_stepsizes=update_stepsizes,
                      replace_weights=update_weights,
                      log_density=gmm.log_density,
                      average_entropy=gmm.average_entropy,
                      component_log_densities=gmm.component_log_densities,
                      log_densities_also_individual=gmm.log_densities_also_individual,
                      replace_components=gmm.replace_components,
                      sample_from_components_no_shuffle=gmm.sample_from_components_no_shuffle,
                      log_density_and_grad=gmm.log_density_and_grad,
                      sample=gmm.sample)
