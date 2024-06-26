import jax
import jax.numpy as jnp

from algorithms.scld import resampling
from jax.scipy.special import logsumexp

from algorithms.scld.scld_utils import sample_kernel, log_prob_kernel


def per_sample_sub_traj_is_weight(key, x_start, model_state, params, sim_tuple, sub_traj, forward=True,
                                  stop_grad=False):
    """
    Computes the incremental importance sampling weights for a single sample x.
    """
    (log_density_per_step, noise_schedule, total_steps) = sim_tuple

    dt = 1. / total_steps

    def is_weight_forward(state, step):
        """
        Takes samples from π_{t} and moves them to π_{t+1}. Computes the incremental IS weight.
        """
        x, log_w, key_gen = state

        step = step.astype(jnp.float32)

        # Compute SDE components
        sigma_t = noise_schedule(step)
        scale = sigma_t * jnp.sqrt(2 * dt)
        langevin = jax.grad(log_density_per_step, 1)(step, x)
        langevin_detached = jax.lax.stop_gradient(langevin)
        model_output = model_state.apply_fn(params, x, step * jnp.ones(1), langevin_detached)

        # Euler-Maruyama integration of the SDE
        fwd_mean = x + sigma_t ** 2 * (langevin + model_output) * dt

        key, key_gen = jax.random.split(key_gen)
        x_new = sample_kernel(key, jax.lax.stop_gradient(fwd_mean) if stop_grad else fwd_mean, scale)

        langevin_new = jax.grad(log_density_per_step, 1)(step, x_new)
        langevin_new_detached = jax.lax.stop_gradient(langevin_new)
        model_output_new = model_state.apply_fn(params, x_new, (step + 1) * jnp.ones(1), langevin_new_detached)

        bwd_mean = x_new + sigma_t ** 2 * (langevin_new - model_output_new) * dt

        fwd_log_prob = log_prob_kernel(x_new, fwd_mean, scale)
        bwd_log_prob = log_prob_kernel(x, bwd_mean, scale)

        key, key_gen = jax.random.split(key_gen)
        log_w += bwd_log_prob - fwd_log_prob
        next_state = (x_new, log_w, key_gen)
        per_step_output = (x_new,)
        return next_state, per_step_output

    def is_weight_backward(state, step):
        """
        Takes samples from π_{t+1} and moves them to π_{t}. Computes the incremental IS weight.
        """
        x, log_w, key_gen = state
        next_step = step + 1

        step = step.astype(jnp.float32)
        if stop_grad:
            x = jax.lax.stop_gradient(x)

        # Compute SDE components
        sigma_t = noise_schedule(next_step)
        scale = sigma_t * jnp.sqrt(2 * dt)
        langevin = jax.grad(log_density_per_step, 1)(next_step, x)
        langevin_detached = jax.lax.stop_gradient(langevin)
        model_output = model_state.apply_fn(params, x, next_step * jnp.ones(1), langevin_detached)

        # Euler-Maruyama integration of the SDE
        bwd_mean = x + sigma_t ** 2 * (langevin - model_output) * dt

        key, key_gen = jax.random.split(key_gen)
        x_new = sample_kernel(key, bwd_mean, scale)

        if stop_grad:
            x_new = jax.lax.stop_gradient(x_new)

        langevin_new = jax.grad(log_density_per_step, 1)(next_step, x_new)
        langevin_new_detached = jax.lax.stop_gradient(langevin_new)
        model_output_new = model_state.apply_fn(params, x_new, step * jnp.ones(1), langevin_new_detached)
        fwd_mean = x_new + sigma_t ** 2 * (langevin_new + model_output_new) * dt

        fwd_log_prob = log_prob_kernel(x, fwd_mean, scale)
        bwd_log_prob = log_prob_kernel(x_new, bwd_mean, scale)

        key, key_gen = jax.random.split(key_gen)
        log_w += bwd_log_prob - fwd_log_prob
        next_state = (x_new, log_w, key_gen)
        per_step_output = (x_new,)
        return next_state, per_step_output

    traj_start, traj_end, traj_idx, traj_length = sub_traj

    rng_key, rng_key_gen = jax.random.split(key)
    initial_state = (x_start, 0, rng_key_gen)

    if forward:
        final_state, _ = jax.lax.scan(is_weight_forward, initial_state, traj_start + jnp.arange(traj_length))
        x_final, delta_w, _ = final_state
        final_log_prob, init_log_prob = log_density_per_step(traj_end, x_final), log_density_per_step(traj_start,
                                                                                                      x_start)
    else:
        final_state, _ = jax.lax.scan(is_weight_backward, initial_state, jnp.arange(traj_length)[::-1])
        x_final, delta_w, _ = final_state
        final_log_prob, init_log_prob = log_density_per_step(traj_end, x_start), log_density_per_step(traj_start,
                                                                                                      x_final)

    delta_w += final_log_prob - init_log_prob

    return delta_w, (x_final, final_log_prob)


def sub_traj_is_weights(keys, samples, model_state, params, sim_tuple, sub_traj,
                        forward=True, stop_grad=True):
    """
    Computes the incremental importance weights of a sub-trajectory, i.e.,
    G(x_t, x_t+1) = γ(x_{t+1}) B(x_{t}|x_{t+1}) / γ(x_{t}) F(x_{t+1}|x_{t})
    """
    w, aux = jax.vmap(per_sample_sub_traj_is_weight,
                      in_axes=(0, 0, None, None, None, None, None, None))(
        keys, samples, model_state, params, sim_tuple, sub_traj, forward, stop_grad)
    return w.reshape(-1, ), aux


def get_lnz_elbo_increment(log_is_weights, log_weights):
    normalized_log_weights = jax.nn.log_softmax(log_weights)
    total_terms = normalized_log_weights + log_is_weights
    lnz_inc = logsumexp(total_terms)
    elbo_inc = jnp.sum(jnp.exp(normalized_log_weights) * log_is_weights)
    return lnz_inc, elbo_inc


def update_samples_log_weights(
        samples,
        log_is_weights,
        log_weights,
        markov_kernel_apply,
        key,
        step: int,
        use_reweighting: bool,
        use_resampling: bool,
        resampler,
        use_markov: bool,
        resample_threshold: float):
    if use_reweighting:
        log_weights_new = reweight(log_weights, log_is_weights)
    else:
        log_weights_new = log_weights

    if use_resampling:
        subkey, key = jax.random.split(key)
        resampled_samples, log_weights_resampled = resampling.optionally_resample(
            subkey, log_weights_new, samples, resample_threshold, resampler)
    else:
        resampled_samples = samples
        log_weights_resampled = log_weights_new

    if use_markov:
        markov_samples, acceptance_tuple = markov_kernel_apply(step, key, resampled_samples)
    else:
        markov_samples = resampled_samples
        acceptance_tuple = (1., 1., 1.)

    return markov_samples, log_weights_resampled, acceptance_tuple


def reweight(log_weights_old, log_is_weights):
    log_weights_new_unorm = log_weights_old + log_is_weights
    log_weights_new = jax.nn.log_softmax(log_weights_new_unorm)
    return log_weights_new
