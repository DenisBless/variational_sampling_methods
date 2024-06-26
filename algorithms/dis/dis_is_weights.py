import jax
import jax.numpy as jnp
from functools import partial

from algorithms.cmcd.cmcd_rnd import sample_kernel, log_prob_kernel


def per_sample_rnd(seed, model_state, params, aux_tuple, target, num_steps, noise_schedule,
                   stop_grad=False, prior_to_target=True):
    init_std, init_sampler, init_log_prob = aux_tuple
    target_log_prob = target.log_prob

    def langevin_init_fn(x, t, sigma_t, T, initial_log_prob, target_log_prob):
        tr = t / T
        return sigma_t * ((1 - tr) * target_log_prob(x) + tr * initial_log_prob(x))

    betas = noise_schedule
    langevin_init = partial(langevin_init_fn, T=num_steps, initial_log_prob=init_log_prob,
                            target_log_prob=target_log_prob)
    dt = 1. / num_steps

    def simulate_prior_to_target(state, per_step_input):
        x, log_w, key_gen = state
        step = per_step_input

        step = step.astype(jnp.float32)
        if stop_grad:
            x = jax.lax.stop_gradient(x)

        # Compute SDE components
        beta_t = betas(step)
        sigma_t = jnp.sqrt(2 * beta_t) * init_std
        langevin = jax.lax.stop_gradient(jax.grad(langevin_init)(x, step, sigma_t))
        model_output = model_state.apply_fn(params, x, step * jnp.ones(1), langevin)

        fwd_mean = x + (sigma_t * model_output + beta_t * x) * dt
        scale = sigma_t * jnp.sqrt(dt)
        # Euler-Maruyama integration of the SDE
        key, key_gen = jax.random.split(key_gen)
        x_new = sample_kernel(key, fwd_mean, scale)

        if stop_grad:
            x_new = jax.lax.stop_gradient(x_new)

        bwd_mean = x_new - beta_t * x_new * dt

        fwd_log_prob = log_prob_kernel(x_new, fwd_mean, scale)
        bwd_log_prob = log_prob_kernel(x, bwd_mean, scale)

        key, key_gen = jax.random.split(key_gen)
        log_w += bwd_log_prob - fwd_log_prob
        next_state = (x_new, log_w, key_gen)
        per_step_output = (x_new,)
        return next_state, per_step_output

    def simulate_target_to_prior(state, per_step_input):
        x, log_w, key_gen = state
        step = per_step_input

        step = step.astype(jnp.float32)
        if stop_grad:
            x = jax.lax.stop_gradient(x)

        # Compute SDE components
        beta_t = betas(step)
        sigma_t = jnp.sqrt(2 * beta_t) * init_std
        scale = sigma_t * jnp.sqrt(dt)

        bwd_mean = x - beta_t * x * dt
        # Euler-Maruyama integration of the SDE
        key, key_gen = jax.random.split(key_gen)
        x_new = sample_kernel(key, bwd_mean, scale)

        if stop_grad:
            x_new = jax.lax.stop_gradient(x_new)

        langevin = jax.lax.stop_gradient(jax.grad(langevin_init)(x_new, step, sigma_t))
        model_output = model_state.apply_fn(params, x, step * jnp.ones(1), langevin)

        fwd_mean = (sigma_t * model_output + beta_t * x) * dt

        fwd_log_prob = log_prob_kernel(x, fwd_mean, scale)
        bwd_log_prob = log_prob_kernel(x_new, bwd_mean, scale)

        key, key_gen = jax.random.split(key_gen)
        log_w += bwd_log_prob - fwd_log_prob
        next_state = (x_new, log_w, key_gen)
        per_step_output = (x_new,)
        return next_state, per_step_output

    key, key_gen = jax.random.split(seed)
    if prior_to_target:
        init_x = jnp.clip(init_sampler(seed=key), -4 * init_std, 4 * init_std)
        dim = init_x.shape[0]
        aux = (init_x, 0., key)
        aux, per_step_output = jax.lax.scan(simulate_prior_to_target, aux, jnp.arange(1, num_steps + 1)[::-1])
        final_x, log_ratio, _ = aux
        terminal_cost = init_log_prob(init_x) - target_log_prob(final_x)
    else:
        init_x = target.sample(key, ())
        dim = init_x.shape[0]
        key, key_gen = jax.random.split(key_gen)
        aux = (init_x, 0., key)
        aux, per_step_output = jax.lax.scan(simulate_target_to_prior, aux, jnp.arange(1, num_steps + 1))
        final_x, log_ratio, _ = aux
        terminal_cost = init_log_prob(final_x) - target_log_prob(init_x)

    running_cost = -log_ratio
    x_t = per_step_output
    stochastic_costs = jnp.zeros_like(running_cost)
    return final_x, running_cost, stochastic_costs, terminal_cost, x_t


def rnd(key, model_state, params, batch_size, aux_tuple, target, num_steps, noise_schedule,
        stop_grad=False, prior_to_target=True):
    seeds = jax.random.split(key, num=batch_size)
    x_0, running_costs, stochastic_costs, terminal_costs, x_t = jax.vmap(per_sample_rnd,
                                                              in_axes=(
                                                                  0, None, None, None, None, None, None,
                                                                  None, None)) \
        (seeds, model_state, params, aux_tuple, target, num_steps, noise_schedule, stop_grad, prior_to_target)

    return x_0, running_costs, stochastic_costs, terminal_costs


def neg_elbo(key, model_state, params, batch_size, aux_tuple, target, num_steps, noise_schedule,
             stop_grad=False):
    aux = rnd(key, model_state, params, batch_size, aux_tuple, target, num_steps, noise_schedule,
              stop_grad)
    samples, running_costs, stochastic_costs, terminal_costs = aux
    neg_elbo = running_costs + terminal_costs
    return jnp.mean(neg_elbo), (neg_elbo, samples)
