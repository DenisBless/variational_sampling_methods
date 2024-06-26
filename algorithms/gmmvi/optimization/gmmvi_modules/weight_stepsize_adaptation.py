from typing import NamedTuple, Callable, Optional
import chex
import jax.numpy as jnp
from jax import lax

from algorithms.gmmvi.models.gmm_wrapper import GMMWrapperState
import jax


class WeightStepsizeAdaptationState(NamedTuple):
    stepsize: chex.Array
    DECAYING_num_weight_updates: Optional[chex.Array] = None
    IMPROVEMENT_elbo_history: Optional[chex.Array] = None


class WeightStepsizeAdaptation(NamedTuple):
    init_weight_stepsize_adaptation: Callable
    update_stepsize: Callable


def setup_fixed_weight_stepsize_adaptation():
    def init_fixed_weight_stepsize_adaptation_state(initial_stepsize: jnp.float32):
        return WeightStepsizeAdaptationState(stepsize=jnp.array(initial_stepsize, dtype=jnp.float32))

    def update_stepsize(weight_stepsize_adaption_state: WeightStepsizeAdaptationState, gmm_wrapper_state: GMMWrapperState):
        return weight_stepsize_adaption_state

    return WeightStepsizeAdaptation(init_weight_stepsize_adaptation=init_fixed_weight_stepsize_adaptation_state,
                                    update_stepsize=update_stepsize)


def setup_decaying_weight_stepsize_adaptation(INITIAL_STEPSIZE_CONST: float, ANNEALING_EXPONENT: float):
    def init_decaying_weight_stepsize_adaptation_state(initial_stepsize: jnp.float32):
        return WeightStepsizeAdaptationState(stepsize=initial_stepsize,
                                             DECAYING_num_weight_updates=jnp.array(0, dtype=jnp.float32))

    @jax.jit
    def update_stepsize(weight_stepsize_adaption_state: WeightStepsizeAdaptationState, gmm_wrapper_state: GMMWrapperState):
        return WeightStepsizeAdaptationState(stepsize=INITIAL_STEPSIZE_CONST / (1. + jax.lax.pow(weight_stepsize_adaption_state.DECAYING_num_weight_updates, ANNEALING_EXPONENT)),
                                             DECAYING_num_weight_updates=weight_stepsize_adaption_state.DECAYING_num_weight_updates + 1)

    return WeightStepsizeAdaptation(init_weight_stepsize_adaptation=init_decaying_weight_stepsize_adaptation_state,
                                    update_stepsize=update_stepsize)


def setup_improvement_based_weight_stepsize_adaptation(MIN_STEPSIZE, MAX_STEPSIZE, STEPSIZE_INC_FACTOR, STEPSIZE_DEC_FACTOR):
    def init_improvement_based_weight_stepsize_adaptation_state(initial_stepsize: jnp.float32):
        return WeightStepsizeAdaptationState(stepsize=initial_stepsize,
                                             IMPROVEMENT_elbo_history=jnp.array([jnp.finfo(jnp.float32).min], dtype=jnp.float32))

    @jax.jit
    def update_stepsize(weight_stepsize_adaption_state: WeightStepsizeAdaptationState, gmm_wrapper_state: GMMWrapperState):
        elbo = jnp.sum(jnp.exp(gmm_wrapper_state.gmm_state.log_weights) * gmm_wrapper_state.reward_history[:, -1]) - jnp.sum(
            jnp.exp(gmm_wrapper_state.gmm_state.log_weights) * gmm_wrapper_state.gmm_state.log_weights)

        elbo_history = jnp.concatenate((weight_stepsize_adaption_state.IMPROVEMENT_elbo_history, jnp.expand_dims(elbo, 0)), axis=0)

        stepsize = lax.cond(elbo_history[-1] > elbo_history[-2],
                            lambda stepsize: jnp.minimum(STEPSIZE_INC_FACTOR * stepsize, MAX_STEPSIZE),
                            lambda stepsize: jnp.maximum(STEPSIZE_DEC_FACTOR * stepsize, MIN_STEPSIZE), weight_stepsize_adaption_state.stepsize)
        return WeightStepsizeAdaptationState(stepsize=stepsize, IMPROVEMENT_elbo_history=elbo_history)

    return WeightStepsizeAdaptation(init_weight_stepsize_adaptation=init_improvement_based_weight_stepsize_adaptation_state,
                                    update_stepsize=update_stepsize)
