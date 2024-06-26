from typing import NamedTuple, Callable
import chex
import jax.numpy as jnp
import jax
from algorithms.gmmvi.models.gmm_wrapper import GMMWrapperState


class ComponentStepsizeAdaptationState(NamedTuple):
    pass


class ComponentStepsizeAdaptation(NamedTuple):
    init_component_stepsize_adaptation_state: Callable
    update_stepsize: Callable


def setup_fixed_component_stepsize_adaptation():
    def init_fixed_component_stepsize_adaptation_state():
        return ComponentStepsizeAdaptationState()

    def update_stepsize(gmm_wrapper_state: GMMWrapperState) -> chex.Array:
        return gmm_wrapper_state.stepsizes

    return ComponentStepsizeAdaptation(init_component_stepsize_adaptation_state=init_fixed_component_stepsize_adaptation_state,
                                       update_stepsize=update_stepsize)


def setup_decaying_component_stepsize_adaptation(INITIAL_STEPSIZE: float, ANNEALING_EXPONENT: float):
    def init_decaying_component_stepsize_adaptation_state():
        return ComponentStepsizeAdaptationState()

    def update_stepsize(gmm_wrapper_state: GMMWrapperState) -> chex.Array:

        new_stepsizes = jnp.empty((gmm_wrapper_state.gmm_state.num_components,), dtype=jnp.float32)
        for i in range(jnp.shape(gmm_wrapper_state.stepsizes)[0]):
            new_stepsize = INITIAL_STEPSIZE / (1 + jax.lax.pow(float(gmm_wrapper_state.num_received_updates[i]), ANNEALING_EXPONENT))
            new_stepsizes = new_stepsizes.at[i].set(new_stepsize)

        return jnp.stack(new_stepsizes)

    return ComponentStepsizeAdaptation(init_component_stepsize_adaptation_state=init_decaying_component_stepsize_adaptation_state,
                                       update_stepsize=update_stepsize)


def setup_improvement_based_stepsize_adaptation(MIN_STEPSIZE: float,
                                                MAX_STEPSIZE: float,
                                                STEPSIZE_INC_FACTOR: float,
                                                STEPSIZE_DEC_FACTOR: float):
    def init_improvement_based_component_stepsize_adaptation():
        return ComponentStepsizeAdaptationState()

    @jax.jit
    def update_stepsize(gmm_wrapper_state: GMMWrapperState) -> chex.Array:

        def update_fn(reward_history, current_stepsize):
            return jax.lax.cond(reward_history[-2] >= reward_history[-1],
                                lambda current_stepsize: jnp.maximum(STEPSIZE_DEC_FACTOR * current_stepsize, MIN_STEPSIZE),
                                lambda current_stepsize: jnp.minimum(STEPSIZE_INC_FACTOR * current_stepsize, MAX_STEPSIZE),
                                current_stepsize)

        return jax.vmap(update_fn)(gmm_wrapper_state.reward_history, gmm_wrapper_state.stepsizes)

    return ComponentStepsizeAdaptation(init_component_stepsize_adaptation_state=init_improvement_based_component_stepsize_adaptation,
                                       update_stepsize=update_stepsize)
