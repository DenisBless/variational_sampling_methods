"""Code builds on https://github.com/lollcat/fab-jax"""
from typing import Callable, NamedTuple, Tuple

import chex
import jax.numpy as jnp
import jax.random
import optax

from algorithms.fab.sampling.smc import SequentialMonteCarloSampler, SMCState
from algorithms.fab.flow.flow import Flow, FlowParams

Params = chex.ArrayTree
LogProbFn = Callable[[chex.Array], chex.Array]
ParameterizedLogProbFn = Callable[[chex.ArrayTree, chex.Array], chex.Array]
Info = dict


def reverse_kl_loss(params: chex.ArrayTree,
                    q_sample_and_log_prob_apply,
                    log_q_fn_apply: ParameterizedLogProbFn,
                    log_p_fn: LogProbFn,
                    batch_size: int,
                    path_gradient: bool = True,
                    ):
    x, log_q = q_sample_and_log_prob_apply(params, (batch_size,))
    if path_gradient:
        log_q = log_q_fn_apply(jax.lax.stop_gradient(params), x)
    log_p = log_p_fn(x)
    kl = jnp.mean(log_q - log_p)
    return kl



def fab_loss_smc_samples(params: chex.ArrayTree, x: chex.Array, log_w: chex.Array, log_q_fn_apply: ParameterizedLogProbFn):
    """Estimate FAB loss with a batch of samples from smc."""
    chex.assert_rank(log_w, 1)
    chex.assert_rank(x, 2)

    log_q = log_q_fn_apply(params, x)
    chex.assert_equal_shape((log_q, log_w))
    return - jnp.mean(jax.nn.softmax(log_w) * log_q)


class TrainStateNoBuffer(NamedTuple):
    flow_params: FlowParams
    key: chex.PRNGKey
    opt_state: optax.OptState
    smc_state: SMCState


def build_fab_no_buffer_init_step_fns(flow: Flow, log_p_fn: LogProbFn,
                                      smc: SequentialMonteCarloSampler, optimizer: optax.GradientTransformation,
                                      batch_size: int):

    def init(key: chex.PRNGKey) -> TrainStateNoBuffer:
        """Initialise the flow, optimizer and smc states."""
        key1, key2, key3 = jax.random.split(key, 3)
        dummy_sample = jnp.zeros(flow.dim)
        flow_params = flow.init(key1, dummy_sample)
        opt_state = optimizer.init(flow_params)
        smc_state = smc.init(key2)

        return TrainStateNoBuffer(flow_params=flow_params, key=key3, opt_state=opt_state, smc_state=smc_state)

    @jax.jit
    def step(state: TrainStateNoBuffer) -> Tuple[TrainStateNoBuffer, Info]:
        key, subkey = jax.random.split(state.key)
        info = {}

        # Run smc.
        def log_q_fn(x: chex.Array) -> chex.Array:
            return flow.log_prob_apply(state.flow_params, x)

        x0 = flow.sample_apply(state.flow_params, subkey, (batch_size,))
        chex.assert_rank(x0, 2)  # Currently written assuming x only has 1 event dimension.
        point, log_w, smc_state, smc_info = smc.step(x0, state.smc_state, log_q_fn, log_p_fn)
        info.update(smc_info)

        # Estimate loss and update flow params.
        loss, grad = jax.value_and_grad(fab_loss_smc_samples)(state.flow_params, point.x, log_w, flow.log_prob_apply)
        updates, new_opt_state = optimizer.update(grad, state.opt_state, params=state.flow_params)
        new_params = optax.apply_updates(state.flow_params, updates)
        info.update(loss=loss)

        new_state = TrainStateNoBuffer(flow_params=new_params, key=key, opt_state=new_opt_state, smc_state=smc_state)
        return new_state, info

    return init, step
