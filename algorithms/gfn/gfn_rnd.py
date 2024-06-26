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


def per_sample_rnd(seed, model_state, params, sde_tuple, target, num_steps, noise_schedule, stop_grad=False,
                   prior_to_target=True):
    dim, ref_log_prob = sde_tuple
    target_log_prob = target.log_prob

    def langevin_init_fn(x, t, T, target_log_prob):
        tr = t / T
        return tr * target_log_prob(x)

    sigmas = noise_schedule
    langevin_init = partial(langevin_init_fn, T=num_steps, target_log_prob=target_log_prob)
    dt = 1. / num_steps

    def simulate_prior_to_target(state, per_step_input):
        x, log_w, key_gen = state
        step = per_step_input
        t = step / num_steps
        shrink = (t - dt) / t

        step = step.astype(jnp.float32)
        if stop_grad:
            x = jax.lax.stop_gradient(x)

        # Compute SDE components
        sigma_t = sigmas(step)
        langevin = jax.lax.stop_gradient(jax.grad(langevin_init)(x, step))
        model_output = model_state.apply_fn(params, x, step * jnp.ones(1), langevin)

        # Euler-Maruyama integration of the SDE
        fwd_mean = x + sigma_t * model_output * dt
        fwd_scale = sigma_t * jnp.sqrt(dt)

        key, key_gen = jax.random.split(key_gen)
        x_new = sample_kernel(key, fwd_mean, fwd_scale)

        if stop_grad:
            x_new = jax.lax.stop_gradient(x_new)

        bwd_mean = shrink * x_new
        bwd_scale = sigma_t * jnp.sqrt(shrink * dt) + 1e-8

        # Evaluate transition densities
        fk_log_prob = log_prob_kernel(x_new, fwd_mean, fwd_scale)
        bk_log_prob = log_prob_kernel(x, bwd_mean, bwd_scale)

        # Compute importance weight increment
        log_w += bk_log_prob - fk_log_prob

        key, key_gen = jax.random.split(key_gen)
        next_state = (x_new, log_w, key_gen)
        per_step_output = (x_new,)
        return next_state, per_step_output

    def simulate_target_to_prior(state, per_step_input):
        x, log_w, key_gen = state
        step = per_step_input
        t = step / num_steps
        shrink = (t - dt) / t

        step = step.astype(jnp.float32)
        if stop_grad:
            x = jax.lax.stop_gradient(x)

        # Compute SDE components
        sigma_t = sigmas(step)
        langevin = jax.lax.stop_gradient(jax.grad(langevin_init)(x, step))
        model_output = model_state.apply_fn(params, x, step * jnp.ones(1), langevin)

        # Euler-Maruyama integration of the SDE
        fwd_mean = x + sigma_t * model_output * dt
        fwd_scale = sigma_t * jnp.sqrt(dt)

        key, key_gen = jax.random.split(key_gen)
        x_new = sample_kernel(key, fwd_mean, fwd_scale)

        if stop_grad:
            x_new = jax.lax.stop_gradient(x_new)

        bwd_mean = shrink * x_new
        bwd_scale = sigma_t * jnp.sqrt(shrink * dt) + 1e-8

        # Evaluate transition densities
        fk_log_prob = log_prob_kernel(x_new, fwd_mean, fwd_scale)
        bk_log_prob = log_prob_kernel(x, bwd_mean, bwd_scale)

        # Compute importance weight increment
        log_w += bk_log_prob - fk_log_prob

        key, key_gen = jax.random.split(key_gen)
        next_state = (x_new, log_w, key_gen)
        per_step_output = (x_new,)
        return next_state, per_step_output

    key, key_gen = jax.random.split(seed)
    if prior_to_target:
        init_x = jnp.zeros(dim)
        aux = (init_x, jnp.array(0.), key)
        aux, per_step_output = jax.lax.scan(simulate_prior_to_target, aux, jnp.arange(1, num_steps + 1)[::-1])
        final_x, log_ratio, _ = aux
        running_costs = - log_ratio
        stochastic_costs = jnp.zeros_like(log_ratio)
        terminal_cost = - target_log_prob(final_x)
    else:
        init_x = jnp.squeeze(target.sample(key, (1,)))
        key, key_gen = jax.random.split(key_gen)
        aux = (init_x, jnp.array(0.), key)
        aux, per_step_output = jax.lax.scan(simulate_target_to_prior, aux, jnp.arange(1, num_steps + 1))
        final_x, log_ratio, _ = aux
        running_costs = - log_ratio
        stochastic_costs = jnp.zeros_like(log_ratio)
        terminal_cost = - target_log_prob(init_x)

    return final_x, running_costs, stochastic_costs, terminal_cost, _

    # init_x = jnp.zeros(dim)
    # key, key_gen = jax.random.split(seed)
    # aux = (init_x, 0, key)
    # aux, x_t = jax.lax.scan(simulate_prior_to_target, aux, jnp.arange(1, num_steps + 1))
    # final_x, log_w, _ = aux
    # log_w += target_log_prob(final_x)
    # return final_x, log_w, x_t


def rnd(key, model_state, params, batch_size, aux_tuple, target, num_steps, noise_schedule,
        stop_grad=False, prior_to_target=True):
    seeds = jax.random.split(key, num=batch_size)
    x_0, running_costs, stochastic_costs, terminal_costs, _ = jax.vmap(per_sample_rnd,
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


def trajectory_balance(key, model_state, params, batch_size, initial_density, target, num_steps, noise_schedule,
                       stop_grad=True):
    aux = rnd(key, model_state, params, batch_size, initial_density, target, num_steps, noise_schedule,
              stop_grad)
    samples, running_costs, _, terminal_costs = aux
    tb_vals = jnp.mean(jnp.square(running_costs + terminal_costs - params['params']['logZ']))
    return jnp.mean(tb_vals), (tb_vals, samples)


def log_variance(key, model_state, params, batch_size, initial_density, target_density, num_steps, noise_schedule,
                 stop_grad=True):
    aux = rnd(key, model_state, params, batch_size, initial_density, target_density, num_steps, noise_schedule,
              stop_grad)
    samples, log_w, _ = aux
    return jnp.var(log_w), (log_w, samples)
