"""Code builds on https://github.com/lollcat/fab-jax"""
from typing import Tuple, NamedTuple, Optional
from functools import partial

import chex
import jax.numpy as jnp
import jax.random


from algorithms.fab.sampling.base import TransitionOperator, Point, LogProbFn
from algorithms.fab.sampling.mcmc.blackjax_hmc_rewrite import kernel as hmc_kernel, init as hmc_init


class HMCState(NamedTuple):
    key: chex.PRNGKey
    inverse_mass_maxtric: chex.Array
    step_size: chex.Array


def build_blackjax_hmc(
                 dim: int,
                 n_outer_steps: int = 1,
                 n_inner_steps: int = 5,
                 init_step_size: float = 1e-4,
                 adapt_step_size: bool = True,
                 target_p_accept: float = 0.65,
                 step_size_multiplier: float = 1.02,
) -> TransitionOperator:

    one_step = hmc_kernel(divergence_threshold=1000)


    def init(key: chex.PRNGKey) -> HMCState:
        inverse_mass_matrix = jnp.ones(dim)
        return HMCState(key, inverse_mass_matrix, step_size=jnp.array(init_step_size))

    def step(point: Point,
             transition_operator_state: HMCState,
             beta: chex.Array,
             alpha: float,
             log_q_fn: LogProbFn,
             log_p_fn: LogProbFn,
             ) -> \
            Tuple[Point, HMCState, dict]:

        chex.assert_rank(point.x, 2)
        batch_size = point.x.shape[0]

        hmc_state = jax.vmap(hmc_init, in_axes=(0, None, None))(point, beta, alpha)

        def scan_fn(body, xs):
            key = xs
            key_batch = jax.random.split(key, batch_size)
            hmc_state, transition_operator_state = body
            step_fn_partial = partial(one_step,
                                      log_q_fn=log_q_fn,
                                        log_p_fn=log_p_fn,
                                        step_size=transition_operator_state.step_size,
                                        inverse_mass_matrix=transition_operator_state.inverse_mass_maxtric,
                                        num_integration_steps=n_inner_steps)
            hmc_state, info = jax.vmap(step_fn_partial)(key_batch, hmc_state)
            if adapt_step_size:
                step_size = jax.lax.cond(jnp.mean(info.acceptance_rate) > target_p_accept,
                                         lambda step_size: step_size * step_size_multiplier,
                                         lambda step_size: step_size / step_size_multiplier,
                                         transition_operator_state.step_size)
                transition_operator_state = transition_operator_state._replace(step_size=step_size)
            return (hmc_state, transition_operator_state), info


        key, subkey = jax.random.split(transition_operator_state.key)
        (hmc_state, transition_operator_state), infos = jax.lax.scan(
            scan_fn,
            (hmc_state, transition_operator_state),
            jax.random.split(subkey, n_outer_steps))

        # Info for logging
        info = {f"mean_acceptance_rate": jnp.mean(infos.acceptance_rate)}
        info.update(step_size=transition_operator_state.step_size)

        point_kwargs = hmc_state._asdict()
        del(point_kwargs['beta']); del(point_kwargs['alpha'])
        point_kwargs["x"] = point_kwargs["position"]
        del(point_kwargs['position'])
        point = Point(**point_kwargs)

        transition_operator_state = transition_operator_state._replace(key=key)
        return point, transition_operator_state, info

    return TransitionOperator(init, step)
