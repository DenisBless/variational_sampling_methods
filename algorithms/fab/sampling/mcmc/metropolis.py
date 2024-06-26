"""Code builds on https://github.com/lollcat/fab-jax"""
from typing import Tuple, NamedTuple, Optional

import chex
import jax.numpy as jnp
import jax.random


from algorithms.fab.sampling.base import TransitionOperator, Point, LogProbFn, get_intermediate_log_prob, create_point


class MetropolisState(NamedTuple):
    key: chex.PRNGKey
    step_size: chex.Array


def build_metropolis(
                 dim: int,
                 n_steps: int = 1,
                 init_step_size: float = 1.,
                 tune_step_size: bool = True,
                 target_p_accept: float = 0.65,
                 step_size_multiplier: float = 1.02,
) -> TransitionOperator:

    def init(key: chex.PRNGKey) -> MetropolisState:
        return MetropolisState(key, jnp.array(init_step_size))

    def step(point: Point,
             transition_operator_state: MetropolisState,
             beta: chex.Array,
             alpha: float,
             log_q_fn: LogProbFn,
             log_p_fn: LogProbFn,
             ) -> \
            Tuple[Point, MetropolisState, dict]:

        chex.assert_rank(point.x, 2)
        batch_size = point.x.shape[0]

        def one_step(point: Point, key: chex.PRNGKey) -> Tuple[Point, float]:
            chex.assert_rank(point.x, 1)
            key1, key2 = jax.random.split(key)
            new_x = point.x + jax.random.normal(key1, shape=point.x.shape) * transition_operator_state.step_size
            new_point = create_point(new_x, log_q_fn=log_q_fn, log_p_fn=log_p_fn, with_grad=False)
            chex.assert_trees_all_equal_shapes(point, new_point)

            log_p_accept = get_intermediate_log_prob(log_q=new_point.log_q, log_p=new_point.log_p, beta=beta, alpha=alpha) -\
                           get_intermediate_log_prob(log_q=point.log_q, log_p=point.log_p, beta=beta, alpha=alpha)
            log_threshold = - jax.random.exponential(key2)

            accept = (log_p_accept > log_threshold) & jnp.isfinite(new_point.log_q) & jnp.isfinite(new_point.log_p)
            point = jax.lax.cond(accept, lambda p_new, p: p_new, lambda p_new, p: p, new_point, point)
            p_accept = jnp.clip(jnp.exp(log_p_accept), a_max=1)
            return point, p_accept


        def scan_fn(body, xs):
            key = xs
            key_batch = jax.random.split(key, batch_size)
            point = body
            point, p_accept = jax.vmap(one_step)(point, key_batch)
            mean_p_accept = jnp.mean(p_accept)
            return point, mean_p_accept


        key, subkey = jax.random.split(transition_operator_state.key)
        point, mean_p_accept = jax.lax.scan(scan_fn, point, jax.random.split(subkey, n_steps))
        mean_p_accept = jnp.mean(mean_p_accept)

        if tune_step_size:
            step_size = jax.lax.cond(mean_p_accept > target_p_accept,
                                     lambda step_size: step_size*step_size_multiplier,
                                     lambda step_size: step_size / step_size_multiplier,
                                     transition_operator_state.step_size)
            transition_operator_state = transition_operator_state._replace(step_size=step_size)

        # Info for logging
        info = {}
        info.update(mean_p_accept=mean_p_accept, step_size=transition_operator_state.step_size)
        transition_operator_state = transition_operator_state._replace(key=key)
        return point, transition_operator_state, info

    return TransitionOperator(init, step, uses_grad=False)



# from fab.utils.jax_util import broadcasted_where
#
# def build_metropolis(
#                  dim: int,
#                  n_steps: int = 1,
#                  init_step_size: float = 1.,
#                  tune_step_size: bool = True,
#                  target_p_accept: float = 0.65,
#                  step_size_multiplier: float = 1.02,
# ) -> TransitionOperator:
#
#     def init(key: chex.PRNGKey) -> MetropolisState:
#         return MetropolisState(key, jnp.array(init_step_size*dim))
#
#
#     def step(point: Point,
#              transition_operator_state: MetropolisState,
#              beta: chex.Array,
#              alpha: float,
#              log_q_fn: LogProbFn,
#              log_p_fn: LogProbFn,
#              ) -> \
#             Tuple[Point, MetropolisState, dict]:
#
#         chex.assert_rank(point.x, 2)
#         batch_size = point.x.shape[0]
#
#         key, subkey = jax.random.split(transition_operator_state.key)
#
#         for i in range(n_steps):
#             key1, key2 = jax.random.split(key)
#             new_x = point.x + jax.random.normal(key1, shape=point.x.shape) * transition_operator_state.step_size
#             new_point = jax.vmap(create_point, in_axes=(0, None, None, None))(new_x, log_q_fn, log_p_fn, False)
#             chex.assert_trees_all_equal_shapes(point, new_point)
#
#             log_p_accept = get_intermediate_log_prob(log_q=new_point.log_q, log_p=new_point.log_p, beta=beta, alpha=alpha) -\
#                            get_intermediate_log_prob(log_q=point.log_q, log_p=point.log_p, beta=beta, alpha=alpha)
#             log_threshold = - jax.random.exponential(key2, shape=point.x.shape[:1])
#
#             accept = (log_p_accept > log_threshold) & jnp.isfinite(new_point.log_q) & jnp.isfinite(new_point.log_p)
#             point = jax.tree_map(lambda a, b: broadcasted_where(accept, a, b), new_point, point)
#             mean_p_accept = jnp.mean(jnp.clip(jnp.exp(log_p_accept), a_max=1))
#
#         # Info for logging
#         info = {}
#         info.update(mean_p_accept=mean_p_accept, step_size=transition_operator_state.step_size)
#         transition_operator_state = transition_operator_state._replace(key=key)
#         return point, transition_operator_state, info
#
#     return TransitionOperator(init, step, uses_grad=False)
#

