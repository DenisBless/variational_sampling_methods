import jax
import jax.numpy as jnp

from algorithms.scld.is_weights import per_sample_sub_traj_is_weight


def sub_traj_fwd_kl(keys, samples, samples_next, model_state, params, sim_tuple,
                    traj_start, traj_end, traj_idx, traj_length, forward=False, stop_grad=False):
    sub_traj = traj_start, traj_end, traj_idx, traj_length
    w, samples_new = jax.vmap(per_sample_sub_traj_is_weight,
                              in_axes=(0, 0, None, None, None, None, None, None))(
        keys, samples_next, model_state, params, sim_tuple, sub_traj, forward, stop_grad)
    return w.mean(), (w, samples_new)


def sub_traj_rev_kl(keys, samples, samples_next, model_state, params, sim_tuple,
                    traj_start, traj_end, traj_idx, traj_length, forward=True, stop_grad=False):
    sub_traj = traj_start, traj_end, traj_idx, traj_length
    w, samples_new = jax.vmap(per_sample_sub_traj_is_weight,
                              in_axes=(0, 0, None, None, None, None, None, None))(
        keys, samples, model_state, params, sim_tuple, sub_traj, forward, stop_grad)
    return -1. * w.mean(), (w, samples_new)


def sub_traj_fwd_tb(keys, samples, samples_next, model_state, params, sim_tuple,
                    traj_start, traj_end, traj_idx, traj_length, forward=False, stop_grad=True):
    sub_traj = traj_start, traj_end, traj_idx, traj_length
    log_w, samples_new = jax.vmap(per_sample_sub_traj_is_weight,
                                  in_axes=(0, 0, None, None, None, None, None, None))(
        keys, samples_next, model_state, params, sim_tuple, sub_traj, forward, stop_grad)
    tb_vals = jnp.mean(jnp.square(log_w - params['params']['logZ'][traj_idx]))
    return tb_vals, (log_w, samples_new)


def sub_traj_rev_tb(keys, samples, samples_next, model_state, params, sim_tuple,
                    traj_start, traj_end, traj_idx, traj_length, forward=True, stop_grad=True):
    sub_traj = traj_start, traj_end, traj_idx, traj_length
    log_w, samples_new = jax.vmap(per_sample_sub_traj_is_weight,
                                  in_axes=(0, 0, None, None, None, None, None, None))(
        keys, samples, model_state, params, sim_tuple, sub_traj, forward, stop_grad)
    # tb_vals = jnp.mean(jnp.square(log_w - params['params']['logZ'][traj_idx]))
    tb_vals = jnp.mean(jnp.square(log_w))
    return tb_vals, (log_w, samples_new)


def sub_traj_fwd_logvar(keys, samples, samples_next, model_state, params, sim_tuple,
                        traj_start, traj_end, traj_idx, traj_length, forward=False, stop_grad=True):
    sub_traj = traj_start, traj_end, traj_idx, traj_length
    w, samples_new = jax.vmap(per_sample_sub_traj_is_weight,
                              in_axes=(0, 0, None, None, None, None, None, None))(
        keys, samples_next, model_state, params, sim_tuple, sub_traj, forward, stop_grad)
    return jnp.clip(w.var(ddof=0), -1e7, 1e7), (w, samples_new)


def sub_traj_rev_logvar(keys, samples, samples_next, model_state, params, sim_tuple,
                        traj_start, traj_end, traj_idx, traj_length, forward=True, stop_grad=True):
    sub_traj = traj_start, traj_end, traj_idx, traj_length
    w, samples_new = jax.vmap(per_sample_sub_traj_is_weight,
                              in_axes=(0, 0, None, None, None, None, None, None))(
        keys, samples, model_state, params, sim_tuple, sub_traj, forward, stop_grad)
    return jnp.clip(w.var(ddof=0), -1e7, 1e7), (w, samples_new)


# def sub_traj_jd(keys, samples, next_samples, model_state, params, log_density_per_step,
#                 traj_start, traj_end, traj_idx, traj_length, stop_grad=False):
#     sub_traj = traj_start, traj_end, traj_idx, traj_length
#     w_fwd, samples_new = jax.vmap(per_sample_sub_traj_is_weight,
#                                   in_axes=(0, 0, None, None, None, None, None, None, None, None))(
#         keys, samples, model_state, params, log_density_per_step, sub_traj,
#         True, stop_grad)
#
#     w_bwd, samples_bwd = jax.vmap(per_sample_sub_traj_is_weight,
#                                   in_axes=(0, 0, None, None, None, None, None, None, None, None))(
#         keys, next_samples, model_state, params, log_density_per_step, sub_traj,
#         False, stop_grad)
#
#     return w_bwd.mean() - w_fwd.mean(), (w_fwd, samples_new)
#
#
# def sub_traj_js(keys, samples, next_samples, model_state, params, log_density_per_step,
#                 traj_start, traj_end, traj_idx, traj_length, stop_grad=False):
#     sub_traj = traj_start, traj_end, traj_idx, traj_length
#     w_fwd, samples_new = jax.vmap(per_sample_sub_traj_is_weight,
#                                   in_axes=(0, 0, None, None, None, None, None, None, None, None))(
#         keys, samples, model_state, params, log_density_per_step, sub_traj,
#         True, stop_grad)
#
#     w_bwd, samples_bwd = jax.vmap(per_sample_sub_traj_is_weight,
#                                   in_axes=(0, 0, None, None, None, None, None, None, None, None))(
#         keys, next_samples, model_state, params, log_density_per_step, sub_traj,
#         False, stop_grad)
#
#     return -(jax.nn.softplus(-w_bwd).mean() + jax.nn.softplus(w_fwd).mean()) + jnp.log(4), (w_fwd, samples_new)


def get_loss_fn(identifier: str):
    if identifier == 'fwd_kl':
        return sub_traj_fwd_kl
    elif identifier == 'rev_kl':
        return sub_traj_rev_kl
    elif identifier == 'fwd_tb':
        return sub_traj_fwd_tb
    elif identifier == 'rev_tb':
        return sub_traj_rev_tb
    elif identifier == 'fwd_lv':
        return sub_traj_fwd_logvar
    elif identifier == 'rev_lv':
        return sub_traj_rev_logvar
    # elif identifier == 'jd':
    #     return sub_traj_jd
    # elif identifier == 'js':
    #     return sub_traj_js
    else:
        raise ValueError(f'{identifier} is not a valid identifier for a loss function')
