"""Code builds on https://github.com/lollcat/fab-jax"""
from typing import Callable, NamedTuple, Tuple, Optional

import chex
import jax.numpy as jnp
import jax.random
import optax
from jax.flatten_util import ravel_pytree

from algorithms.fab.sampling.smc import SequentialMonteCarloSampler, SMCState
from algorithms.fab.flow.flow import Flow, FlowParams
from algorithms.fab.buffer import PrioritisedBuffer, PrioritisedBufferState
from algorithms.fab.utils.optimize import IgnoreNanOptState

Params = chex.ArrayTree
LogProbFn = Callable[[chex.Array], chex.Array]
ParameterizedLogProbFn = Callable[[chex.ArrayTree, chex.Array], chex.Array]
Info = dict


def fab_loss_buffer_samples_fn(
        params: FlowParams,
        x: chex.Array,
        log_q_old: chex.Array,
        alpha: chex.Array,
        log_q_fn_apply: ParameterizedLogProbFn,
        w_adjust_clip: float,
) -> Tuple[chex.Array, Tuple[chex.Array, chex.Array]]:
    """Estimate FAB loss with a batch of samples from the prioritized replay buffer."""
    chex.assert_rank(x, 2)
    chex.assert_rank(log_q_old, 1)

    log_q = log_q_fn_apply(params, x)
    log_w_adjust = (1 - alpha) * (jax.lax.stop_gradient(log_q) - log_q_old)
    chex.assert_equal_shape((log_q, log_w_adjust))
    w_adjust = jnp.clip(jnp.exp(log_w_adjust), a_max=w_adjust_clip)
    return - jnp.mean(w_adjust * log_q), (log_w_adjust, log_q)


def reverse_kl_loss_fn(params: FlowParams,
                       key: chex.PRNGKey,
                       batch_size: int,
                       sample_and_log_prob_apply: Callable,
                       target_log_prob: LogProbFn,
                       ) -> chex.Array:
    """KL(q||p)."""
    x, log_q = sample_and_log_prob_apply(params, key, (batch_size,))
    log_p = target_log_prob(x)
    kl = jnp.mean(log_q - log_p)
    return kl


def generic_loss(params: FlowParams,
        x: chex.Array,
        key: chex.PRNGKey,
        log_q_old: chex.Array,
        alpha: chex.Array,
        flow: Flow,
        w_adjust_clip: float,
        use_reverse_kl_loss: bool = False,
        target_log_prob: Optional[LogProbFn] = None
        ) -> Tuple[chex.Array, Tuple[chex.Array, chex.Array, dict]]:
    """Generic loss function for training.
    Allows for experimentation for additions to the vanilla FAB loss (e.g. adding a reverse kl loss)."""
    info = {}
    fab_loss, (log_w_adjust, log_q) = fab_loss_buffer_samples_fn(params, x, log_q_old, alpha, flow.log_prob_apply,
                                                                 w_adjust_clip)
    info.update(fab_loss=fab_loss)
    if use_reverse_kl_loss:
        assert target_log_prob is not None
        batch_size = x.shape[0]
        rkld_loss = reverse_kl_loss_fn(params, key, batch_size, flow.sample_and_log_prob_apply, target_log_prob)
        fab_loss = fab_loss + rkld_loss
        info.update(rkld_loss=rkld_loss, total_loss=fab_loss)
    return fab_loss, (log_w_adjust, log_q, info)


class TrainStateWithBuffer(NamedTuple):
    flow_params: FlowParams
    key: chex.PRNGKey
    opt_state: optax.OptState
    smc_state: SMCState
    buffer_state: PrioritisedBufferState


def build_fab_with_buffer_init_step_fns(
        flow: Flow,
        log_p_fn: LogProbFn,
        smc: SequentialMonteCarloSampler,
        buffer: PrioritisedBuffer,
        optimizer: optax.GradientTransformation,
        batch_size: int,
        n_updates_per_smc_forward_pass: int,
        alpha: float = 2.,
        w_adjust_clip: float = 10.,
        use_reverse_kl_loss: bool = False
):
    """Create the `init` and `step` functions that define the FAB algorithm."""
    assert smc.alpha == alpha

    def init(key: chex.PRNGKey) -> TrainStateWithBuffer:
        """Initialise the flow, optimizer, SMC and buffer states."""
        key1, key2, key3, key4 = jax.random.split(key, 4)
        dummy_sample = jnp.zeros(flow.dim)
        flow_params = flow.init(key1, dummy_sample)
        opt_state = optimizer.init(flow_params)
        smc_state = smc.init(key2)

        # Now run multiple forward passes of SMC to fill the buffer. This also
        # tunes the SMC state in the process.
        def log_q_fn(x: chex.Array) -> chex.Array:
            return flow.log_prob_apply(flow_params, x)

        def body_fn(carry, xs):
            """Generate samples with AIS/SMC."""
            smc_state = carry
            key = xs
            x0 = flow.sample_apply(flow_params, key, (batch_size,))
            chex.assert_rank(x0, 2)  # Currently written assuming x only has 1 event dimension.
            point, log_w, smc_state, smc_info = smc.step(x0, smc_state, log_q_fn, log_p_fn)
            return smc_state, (point.x, log_w, point.log_q)

        n_forward_pass = (buffer.min_lengtht_to_sample // batch_size) + 1
        smc_state, (x, log_w, log_q) = jax.lax.scan(body_fn, init=smc_state,
                                                    xs=jax.random.split(key4, n_forward_pass))

        buffer_state = buffer.init(jnp.reshape(x, (n_forward_pass*batch_size, flow.dim)),
                                               log_w.flatten(),
                                               log_q.flatten())

        return TrainStateWithBuffer(flow_params=flow_params, key=key3, opt_state=opt_state,
                                    smc_state=smc_state, buffer_state=buffer_state)

    def one_gradient_update(carry: Tuple[FlowParams, optax.OptState], xs: Tuple[chex.Array, chex.Array,
    chex.PRNGKey]):
        """Perform on update to the flow parameters with a batch of data from the buffer."""
        flow_params, opt_state = carry
        x, log_q_old, key = xs
        info = {}

        # Estimate loss and update flow params.
        grad, (log_w_adjust, log_q, info) = jax.grad(generic_loss, has_aux=True)(
            flow_params, x, key, log_q_old, alpha, flow, w_adjust_clip, use_reverse_kl_loss, log_p_fn)
        updates, new_opt_state = optimizer.update(grad, opt_state, params=flow_params)
        new_params = optax.apply_updates(flow_params, updates)
        grad_norm = optax.global_norm(grad)
        info.update(log10_grad_norm=jnp.log10(grad_norm))  # Makes scale nice for plotting
        info.update(log10_max_param_grad=jnp.log(jnp.max(ravel_pytree(grad)[0])))
        if isinstance(opt_state, IgnoreNanOptState):
            info.update(ignored_grad_count=opt_state.ignored_grads_count,
                        total_optimizer_steps=opt_state.total_steps)
        return (new_params, new_opt_state), (info, log_w_adjust, log_q)


    @jax.jit
    def step(state: TrainStateWithBuffer) -> Tuple[TrainStateWithBuffer, Info]:
        """Perform a single iteration of the FAB algorithm."""
        info = {}

        # Sample from buffer.
        key, subkey = jax.random.split(state.key)
        x_buffer, log_q_old_buffer, indices = buffer.sample_n_batches(subkey, state.buffer_state, batch_size,
                                                                      n_updates_per_smc_forward_pass)
        # Perform sgd steps on flow.
        key, subkey = jax.random.split(key)
        (new_flow_params, new_opt_state), (infos, log_w_adjust, log_q_old) = jax.lax.scan(
            one_gradient_update, init=(state.flow_params, state.opt_state),
            xs=(x_buffer, log_q_old_buffer, jax.random.split(subkey, n_updates_per_smc_forward_pass)),
            length=n_updates_per_smc_forward_pass
        )
        # Adjust samples in the buffer.
        buffer_state = buffer.adjust(log_q=log_q_old.flatten(), log_w_adjustment=log_w_adjust.flatten(),
                                     indices=indices.flatten(),
                                     buffer_state=state.buffer_state)
        # Update info.
        for i in range(n_updates_per_smc_forward_pass):
            info.update(jax.tree_map(lambda x: x[i], infos))

        # Run smc and add samples to the buffer. Note this is done with the flow params before they were updated so that
        # this can occur in parallel (jax will do this after compilation).
        def log_q_fn(x: chex.Array) -> chex.Array:
            return flow.log_prob_apply(state.flow_params, x)

        key, subkey = jax.random.split(key)
        x0 = flow.sample_apply(state.flow_params, subkey, (batch_size,))
        chex.assert_rank(x0, 2)  # Currently written assuming x only has 1 event dimension.
        point, log_w, smc_state, smc_info = smc.step(x0, state.smc_state, log_q_fn, log_p_fn)
        info.update(smc_info)
        buffer_state = buffer.add(x=point.x, log_w=log_w, log_q=point.log_q, buffer_state=buffer_state)

        new_state = TrainStateWithBuffer(flow_params=new_flow_params, key=key, opt_state=new_opt_state,
                                         smc_state=smc_state, buffer_state=buffer_state)
        return new_state, info

    return init, step
