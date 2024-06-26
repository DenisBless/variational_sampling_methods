import jax
import jax.numpy as jnp
from functools import partial


def cos_sq_fn_step_scheme(n_steps, noise_scale=1., s=0.008, dtype=jnp.float32):
    pre_phase = jnp.linspace(0, 1, n_steps, dtype=dtype)
    phase = ((pre_phase + s) / (1 + s)) * jnp.pi * 0.5

    dts = jnp.cos(phase) ** 4
    dts /= dts.sum()
    dts_out = noise_scale * jnp.clip(jnp.concatenate((jnp.array([0]), jnp.cumsum(dts))), 0, 0.999999)
    return dts_out[1:] - dts_out[:-1]


def per_sample_rnd(seed, model_state, params, initial_density_tuple, target, num_steps, noise_schedule,
                   stop_grad=False, prior_to_target=True):
    init_std, init_sampler, init_log_prob, noise_scale = initial_density_tuple
    target_log_prob = target.log_prob

    def langevin_init_fn(x, t, T, initial_log_prob, target_log_prob):
        tr = t / T
        return target_log_prob(x)

    langevin_init = partial(langevin_init_fn, T=num_steps, initial_log_prob=init_log_prob,
                            target_log_prob=target_log_prob)
    betas = cos_sq_fn_step_scheme(num_steps, noise_scale=noise_scale)[::-1]

    def simulate_prior_to_target(state, per_step_input):
        x, key_gen = state
        step = per_step_input
        beta_t = jnp.clip(jnp.sqrt(betas[step]), 0, 1)
        alpha_t = jnp.sqrt(1 - beta_t ** 2)
        step = step.astype(jnp.float32)
        if stop_grad:
            x = jax.lax.stop_gradient(x)

        # Compute SDE components
        langevin = jax.lax.stop_gradient(jax.grad(langevin_init)(x, step))
        model_output = model_state.apply_fn(params, x, step * jnp.ones(1), langevin)
        key, key_gen = jax.random.split(key_gen)
        noise = jnp.clip(jax.random.normal(key, shape=x.shape), -4, 4)

        # Exponential integration of the SDE
        x_new = alpha_t * x + beta_t ** 2 * model_output + beta_t * noise * init_std

        if stop_grad:
            x_new = jax.lax.stop_gradient(x_new)

        # Compute (running) Radon-Nikodym derivative components
        running_cost = 0.5 * beta_t ** 2 * jnp.square(jnp.linalg.norm(model_output)) * (1 / init_std ** 2)
        stochastic_cost = (model_output * noise).sum() * beta_t / init_std

        next_state = (x_new, key_gen)
        per_step_output = (running_cost, stochastic_cost, x_new)
        return next_state, per_step_output

    def simulate_target_to_prior(state, per_step_input):
        x, key_gen = state
        step = per_step_input
        beta_t = jnp.clip(jnp.sqrt(betas[step]), 0, 1)
        alpha_t = jnp.sqrt(1 - beta_t ** 2)
        step = step.astype(jnp.float32)
        if stop_grad:
            x = jax.lax.stop_gradient(x)

        # Compute SDE components
        langevin = jax.lax.stop_gradient(jax.grad(langevin_init)(x, step))
        model_output = model_state.apply_fn(params, x, step * jnp.ones(1), langevin)
        key, key_gen = jax.random.split(key_gen)
        noise = jnp.clip(jax.random.normal(key, shape=x.shape), -4, 4)

        # Exponential integration of the SDE
        x_new = alpha_t * x + beta_t * noise * init_std

        if stop_grad:
            x_new = jax.lax.stop_gradient(x_new)

        # Compute (running) Radon-Nikodym derivative components
        running_cost = -(0.5 * beta_t ** 2 * jnp.square(jnp.linalg.norm(model_output)) * (1 / init_std ** 2))
        stochastic_cost = (model_output * noise).sum() * beta_t / init_std

        next_state = (x_new, key_gen)
        per_step_output = (running_cost, stochastic_cost, x_new)
        return next_state, per_step_output

    key, key_gen = jax.random.split(seed)
    if prior_to_target:
        init_x = jnp.clip(init_sampler(seed=key), -4 * init_std, 4 * init_std)
        key, key_gen = jax.random.split(key_gen)
        aux = (init_x, key)
        aux, per_step_output = jax.lax.scan(simulate_prior_to_target, aux, jnp.arange(1, num_steps + 1)[::-1])
        final_x, _ = aux
        terminal_cost = init_log_prob(final_x) - target_log_prob(final_x)
    else:
        init_x = jnp.squeeze(target.sample(key, (1,)))
        key, key_gen = jax.random.split(key_gen)
        aux = (init_x, key)
        aux, per_step_output = jax.lax.scan(simulate_target_to_prior, aux, jnp.arange(1, num_steps + 1))
        final_x, _ = aux
        terminal_cost = init_log_prob(init_x) - target_log_prob(init_x)

    running_cost, stochastic_cost, x_t = per_step_output
    return final_x, running_cost, stochastic_cost, terminal_cost, x_t


def rnd(key, model_state, params, batch_size, initial_density_tuple, target, num_steps, noise_schedule,
        stop_grad=False, prior_to_target=True):
    seeds = jax.random.split(key, num=batch_size)
    x_0, running_costs, stochastic_costs, terminal_costs, x_t = jax.vmap(per_sample_rnd,
                                                                         in_axes=(
                                                                             0, None, None, None, None, None, None,
                                                                             None, None)) \
        (seeds, model_state, params, initial_density_tuple, target, num_steps, noise_schedule, stop_grad,
         prior_to_target)

    return x_0, running_costs.sum(1), stochastic_costs.sum(1), terminal_costs


def neg_elbo(key, model_state, params, batch_size, initial_density, target_density, num_steps, noise_schedule,
             stop_grad=False):
    aux = rnd(key, model_state, params, batch_size, initial_density, target_density, num_steps, noise_schedule,
              stop_grad)
    samples, running_costs, _, terminal_costs = aux
    neg_elbo_vals = running_costs + terminal_costs
    return jnp.mean(neg_elbo_vals), (neg_elbo_vals, samples)