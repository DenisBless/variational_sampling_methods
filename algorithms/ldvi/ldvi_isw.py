import jax
import jax.numpy as jnp
from functools import partial

import numpyro.distributions as npdist


def sample_kernel(rng_key, mean, scale):
    eps = jax.random.normal(rng_key, shape=(mean.shape[0],))
    return mean + scale * eps


def log_prob_kernel(x, mean, scale):
    dist = npdist.Independent(npdist.Normal(loc=mean, scale=scale), 1)
    return dist.log_prob(x)


def per_sample_rnd(seed, model_state, params, aux_tuple, target, num_steps, noise_schedule,
                   stop_grad=False, prior_to_target=True):
    prior_sampler, prior_log_prob, get_betas, get_diff_coefficient, get_friction = aux_tuple
    target_log_prob = target.log_prob

    def langevin_score_fn(x, beta, params, initial_log_prob, target_log_prob):
        return (beta * target_log_prob(x) + (1 - beta) * initial_log_prob(params, x))

    deltas = noise_schedule
    betas = get_betas(params)

    langevin_score = partial(langevin_score_fn, initial_log_prob=prior_log_prob, target_log_prob=target_log_prob)

    def simulate_prior_to_target(state, per_step_input):
        """
        Takes samples from the prior and moves them to the target
        """

        x, rho, log_w, key_gen = state
        step = per_step_input

        step = step.astype(jnp.float32)
        if stop_grad:
            x = jax.lax.stop_gradient(x)

        # Compute SDE components
        eps = deltas(step) * jnp.square(get_diff_coefficient(params))
        beta_t = betas(step)
        friction = get_friction(params)
        eta = eps * friction
        scale = jnp.sqrt(2 * eta)

        # Forward kernel
        fwd_rho_mean = rho * (1 - eta)
        key, key_gen = jax.random.split(key_gen)
        rho_prime = sample_kernel(key, fwd_rho_mean, scale)

        # Leapfrog integration
        langevin = jax.grad(langevin_score)(x, beta_t, params)
        rho_prime_prime = rho_prime + eps * langevin / 2.0
        x_new = x + eps * rho_prime_prime
        langevin_new = jax.grad(langevin_score)(x_new, beta_t, params)
        rho_new = rho_prime_prime + eps * langevin_new / 2.0

        # Backward kernel
        model_input = jnp.concatenate([x, rho_prime])
        bwd_rho_mean = rho_prime * (1.0 - eta) + 2 * eta * model_state.apply_fn(params, model_input, step, jax.lax.stop_gradient(langevin_new))

        # Evaluate kernels
        fwd_log_prob = log_prob_kernel(rho_prime, fwd_rho_mean, scale)
        bwd_log_prob = log_prob_kernel(rho, bwd_rho_mean, scale)

        # Update weight and return
        log_w += bwd_log_prob - fwd_log_prob

        key, key_gen = jax.random.split(key_gen)
        next_state = (x_new, rho_new, log_w, key_gen)
        per_step_output = (x_new,)
        return next_state, per_step_output

    def simulate_target_to_prior(state, per_step_input):
        """
        Takes samples from the target and moves them to the prior
        """
        x, rho, log_w, key_gen = state
        step = per_step_input

        step = step.astype(jnp.float32)
        if stop_grad:
            x = jax.lax.stop_gradient(x)

        # Compute SDE components
        eps = deltas(step) * jnp.square(get_diff_coefficient(params))
        beta_t = betas(step)
        friction = get_friction(params)
        eta = eps * friction
        scale = jnp.sqrt(2 * eta)

        # Backward kernel
        # Backward kernel
        model_input = jnp.concatenate([x, rho])
        langevin = jax.grad(langevin_score)(x, beta_t, params)
        bwd_rho_mean = rho * (1.0 - eta) + 2 * eta * model_state.apply_fn(params, model_input, step, jax.lax.stop_gradient(langevin))
        key, key_gen = jax.random.split(key_gen)
        rho_prime = sample_kernel(key, bwd_rho_mean, scale)

        # Leapfrog integration
        rho_prime_prime = rho_prime - eps * langevin / 2.0
        x_new = x - eps * rho_prime_prime
        langevin_new = jax.grad(langevin_score)(x_new, beta_t, params)
        rho_new = rho_prime_prime - eps * langevin_new / 2.0

        # Forward kernel
        fwd_rho_mean = rho_prime * (1.0 - eta)

        # Evaluate kernels
        fwd_log_prob = log_prob_kernel(rho, fwd_rho_mean, scale)
        bwd_log_prob = log_prob_kernel(rho_prime, bwd_rho_mean, scale)

        # Update weight and return
        log_w += bwd_log_prob - fwd_log_prob

        key, key_gen = jax.random.split(key_gen)
        next_state = (x_new, rho_new, log_w, key_gen)
        per_step_output = (x_new,)
        return next_state, per_step_output

    key, key_gen = jax.random.split(seed)
    key, key_gen = jax.random.split(key_gen)

    if prior_to_target:
        init_x = jnp.squeeze(prior_sampler(params, key, 1))
        key, key_gen = jax.random.split(key_gen)
        init_rho = jax.random.normal(key, shape=(init_x.shape[0],))  # (dim,)
        aux = (init_x, init_rho, 0., key)
        aux, per_step_output = jax.lax.scan(simulate_prior_to_target, aux, jnp.arange(0, num_steps))
        final_x, final_rho, log_ratio, _ = aux
        sample_terminal_cost = prior_log_prob(params, init_x) - target_log_prob(final_x)
        momentum_terminal_cost = log_prob_kernel(init_rho, jnp.zeros(init_x.shape[0]), 1.0) - log_prob_kernel(final_rho, jnp.zeros(init_x.shape[0]), 1.0)
    else:
        init_x = jnp.squeeze(target.sample(key, (1,)))
        key, key_gen = jax.random.split(key_gen)
        init_rho = jax.random.normal(key, shape=(init_x.shape[0],))  # (dim,)
        aux = (init_x, init_rho, 0., key)
        aux, per_step_output = jax.lax.scan(simulate_target_to_prior, aux, jnp.arange(0, num_steps)[::-1])
        final_x, final_rho, log_ratio, _ = aux
        sample_terminal_cost = prior_log_prob(params, final_x) - target_log_prob(init_x)
        momentum_terminal_cost = log_prob_kernel(final_rho, jnp.zeros(init_x.shape[0]), 1.0) - log_prob_kernel(init_rho, jnp.zeros(init_x.shape[0]), 1.0)

    terminal_cost = sample_terminal_cost + momentum_terminal_cost

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