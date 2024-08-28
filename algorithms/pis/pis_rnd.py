import jax
import jax.numpy as jnp
from functools import partial


def per_sample_rnd(seed, model_state, params, sde_tuple, target, num_steps, noise_schedule, stop_grad=False, prior_to_target=True):
    dim, ref_log_prob = sde_tuple
    target_log_prob = target.log_prob

    def langevin_init_fn(x, t, T, target_log_prob):
        tr = t / T
        return (1 - tr) * target_log_prob(x)

    sigmas = noise_schedule
    langevin_init = partial(langevin_init_fn, T=num_steps, target_log_prob=target_log_prob)
    dt = 1. / num_steps

    def simulate_prior_to_target(state, per_step_input):
        x, sigma_int, key_gen = state
        step = per_step_input

        step = step.astype(jnp.float32)
        if stop_grad:
            x = jax.lax.stop_gradient(x)

        # Compute SDE components
        sigma_t = sigmas(step)
        sigma_int += sigma_t **2 * dt
        langevin = jax.lax.stop_gradient(jax.grad(langevin_init)(x, step))
        model_output = model_state.apply_fn(params, x, step * jnp.ones(1), langevin)
        key, key_gen = jax.random.split(key_gen)
        noise = jnp.clip(jax.random.normal(key, shape=x.shape), -4, 4)

        # Euler-Maruyama integration of the SDE
        x_new = x + sigma_t * model_output * dt + sigma_t * noise * jnp.sqrt(dt)

        if stop_grad:
            x_new = jax.lax.stop_gradient(x_new)

        # Compute (running) Radon-Nikodym derivative components
        running_cost = 0.5 * jnp.square(jnp.linalg.norm(model_output)) * dt
        stochastic_cost = (model_output * noise).sum() * jnp.sqrt(dt)

        next_state = (x_new, sigma_int, key_gen)
        per_step_output = (running_cost, stochastic_cost, x_new)
        return next_state, per_step_output

    def simulate_target_to_prior(state, per_step_input):
        """
        Takes samples from the target and moves them to the prior
        """
        x, sigma_int, key_gen = state
        step = per_step_input

        sigma_t = sigmas(step)
        sigma_int += sigma_t ** 2 * dt
        t = step / num_steps
        shrink = (t - dt) / t

        step = step.astype(jnp.float32)
        if stop_grad:
            x = jax.lax.stop_gradient(x)

        key, key_gen = jax.random.split(key_gen)
        noise = jnp.clip(jax.random.normal(key, shape=x.shape), -4, 4)

        x_new = shrink * x + noise * sigma_t * jnp.sqrt(shrink * dt) + 1e-8

        # Compute SDE components
        langevin = jax.lax.stop_gradient(jax.grad(langevin_init)(x, step))
        model_output = model_state.apply_fn(params, x, step * jnp.ones(1), langevin)

        # Compute (running) Radon-Nikodym derivative components
        running_cost = -0.5 * jnp.square(jnp.linalg.norm(model_output)) * dt
        stochastic_cost = (model_output * noise).sum() * jnp.sqrt(dt)

        next_state = (x_new, sigma_int, key_gen)
        per_step_output = (running_cost, stochastic_cost, x_new)
        return next_state, per_step_output

    key, key_gen = jax.random.split(seed)
    if prior_to_target:
        init_x = jnp.zeros(dim)
        aux = (init_x, jnp.array(0.), key)
        aux, per_step_output = jax.lax.scan(simulate_prior_to_target, aux, jnp.arange(1, num_steps + 1)[::-1])
        final_x, final_sigma, _ = aux
        terminal_cost = ref_log_prob(final_x, jnp.sqrt(final_sigma)) - target_log_prob(final_x)
    else:
        init_x = jnp.squeeze(target.sample(key, (1,)))
        key, key_gen = jax.random.split(key_gen)
        aux = (init_x, jnp.array(0.), key)
        aux, per_step_output = jax.lax.scan(simulate_target_to_prior, aux, jnp.arange(1, num_steps + 1))
        final_x, final_sigma, _ = aux
        terminal_cost = ref_log_prob(init_x, jnp.sqrt(final_sigma)) - target_log_prob(init_x)

    running_cost, stochastic_cost, x_t = per_step_output
    return final_x, running_cost, stochastic_cost, terminal_cost, x_t


def rnd(key, model_state, params, batch_size, initial_density_tuple, target, num_steps, noise_schedule,
        stop_grad=False, prior_to_target=True):
    seeds = jax.random.split(key, num=batch_size)
    x_0, running_costs, stochastic_costs, terminal_costs, x_t = jax.vmap(per_sample_rnd,
                                                                         in_axes=(
                                                                             0, None, None, None, None, None, None,
                                                                             None, None)) \
        (seeds, model_state, params, initial_density_tuple, target, num_steps, noise_schedule, stop_grad, prior_to_target)

    return x_0, running_costs.sum(1), stochastic_costs.sum(1), terminal_costs


def neg_elbo(key, model_state, params, batch_size, initial_density, target_density, num_steps, noise_schedule,
             stop_grad=False):
    aux = rnd(key, model_state, params, batch_size, initial_density, target_density, num_steps, noise_schedule,
              stop_grad)
    samples, running_costs, _, terminal_costs = aux
    neg_elbo_vals = running_costs + terminal_costs
    return jnp.mean(neg_elbo_vals), (neg_elbo_vals, samples)
