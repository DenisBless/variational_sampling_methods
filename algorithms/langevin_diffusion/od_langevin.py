import jax
import jax.numpy as jnp
import algorithms.langevin_diffusion.base_dist as bd
from algorithms.langevin_diffusion.ld_utils import sample_kernel, log_prob_kernel


def evolve_overdamped_langevin(
    x,
    betas,
    params,
    rng_key_gen,
    params_fixed,
    log_prob_model,
    sample_kernel,
    log_prob_kernel,
):
    def U(z, beta):
        return -1.0 * (
            beta * log_prob_model(z) + (1.0 - beta) * bd.log_prob(params["bd"], z)
        )

    def evolve(aux, i):
        x, w, rng_key_gen = aux
        beta = betas[i]

        # Forward kernel
        fk_mean = x - params["eps"] * jax.grad(U)(x, beta)  # - because it is gradient of U = -log \pi
        scale = jnp.sqrt(2 * params["eps"])

        # Sample
        rng_key, rng_key_gen = jax.random.split(rng_key_gen)
        x_new = sample_kernel(rng_key, fk_mean, scale)

        # Backwards kernel
        if alg == 'ULA':  # Unadjusted Langevin Annealing (ULA)
            bk_mean = x_new - params["eps"] * jax.grad(U)(x_new, beta)

        elif alg == 'MCD':  # Monte Carlo Diffusion (MCD)
            bk_mean = (x_new - params["eps"] * jax.grad(U)(x_new, beta)
                       + params["eps"] * apply_fun_approx_network(params["approx_network"], x_new, i))

        else:
            raise ValueError(f'{alg} is not supported.')

        # Evaluate kernels
        fk_log_prob = log_prob_kernel(x_new, fk_mean, scale)
        bk_log_prob = log_prob_kernel(x, bk_mean, scale)

        # Update weight and return
        w += bk_log_prob - fk_log_prob
        rng_key, rng_key_gen = jax.random.split(rng_key_gen)
        aux = x_new, w, rng_key_gen
        return aux, None

    dim, num_temps, apply_fun_approx_network, alg = params_fixed
    # Evolve system
    rng_key, rng_key_gen = jax.random.split(rng_key_gen)
    aux = (x, 0, rng_key_gen)
    aux, _ = jax.lax.scan(evolve, aux, jnp.arange(num_temps))

    x, w, _ = aux
    return x, w, None


def per_sample_elbo(seed, params_flat, unflatten, params_fixed, log_prob):
    params_train, params_notrain = unflatten(params_flat)
    params_notrain = jax.lax.stop_gradient(params_notrain)
    params = {**params_train, **params_notrain}  # Gets all parameters in single place
    dim, num_temps, apply_fun_approx_network, alg = params_fixed

    if num_temps >= 1:
        gridref_y = jnp.cumsum(params["mgridref_y"]) / jnp.sum(params["mgridref_y"])
        gridref_y = jnp.concatenate([jnp.array([0.0]), gridref_y])
        betas = jnp.interp(params["target_x"], params["gridref_x"], gridref_y)
    else:
        raise ValueError('Number of temperatures smaller 1.')

    rng_key_gen = jax.random.PRNGKey(seed)

    rng_key, rng_key_gen = jax.random.split(rng_key_gen)
    x = bd.sample_rep(rng_key, params["bd"])
    w = -bd.log_prob(params["bd"], x)

    if num_temps >= 1:
        rng_key, rng_key_gen = jax.random.split(rng_key_gen)
        x, w_t, _ = evolve_overdamped_langevin(
            x, betas, params, rng_key, params_fixed, log_prob, sample_kernel, log_prob_kernel
        )

        w += w_t

    # Update weight with final model evaluation
    w = w + log_prob(x)
    return -1.0 * w, (x, _)


def compute_elbo(seeds, params_flat, unflatten, params_fixed, log_prob):
    elbos, (x, _) = jax.vmap(per_sample_elbo, in_axes=(0, None, None, None, None))(
        seeds, params_flat, unflatten, params_fixed, log_prob
    )
    return elbos.mean(), (elbos, x)


def evolve_overdamped_langevin_reverse(
        x,
        betas,
        params,
        rng_key_gen,
        params_fixed,
        log_prob_model,
        sample_kernel,
        log_prob_kernel,
):
    def U(z, beta):
        return -1.0 * (
                beta * log_prob_model(z) + (1.0 - beta) * bd.log_prob(params["bd"], z)
        )

    def evolve(aux, i):
        x, w, rng_key_gen = aux
        beta = betas[i]

        # Backwards kernel
        if alg == 'ULA':  # Unadjusted Langevin Annealing (ULA)
            bk_mean = x - params["eps"] * jax.grad(U)(x, beta)

        elif alg == 'MCD':  # Monte Carlo Diffusion (MCD)
            bk_mean = (x - params["eps"] * jax.grad(U)(x, beta)
                       + params["eps"] * apply_fun_approx_network(params["approx_network"], x, i))

        else:
            raise ValueError(f'{alg} is not supported.')

        scale = jnp.sqrt(2 * params["eps"])

        # Sample
        rng_key, rng_key_gen = jax.random.split(rng_key_gen)
        x_new = sample_kernel(rng_key, bk_mean, scale)

        # Forward kernel
        fk_mean = x_new - params["eps"] * jax.grad(U)(x_new, beta)  # - because it is gradient of U = -log \pi

        # Evaluate kernels
        fk_log_prob = log_prob_kernel(x, fk_mean, scale)
        bk_log_prob = log_prob_kernel(x_new, bk_mean, scale)

        # Update weight and return
        w += bk_log_prob - fk_log_prob
        rng_key, rng_key_gen = jax.random.split(rng_key_gen)
        aux = x_new, w, rng_key_gen
        return aux, None

    dim, num_temps, apply_fun_approx_network, alg = params_fixed
    # Evolve system
    rng_key, rng_key_gen = jax.random.split(rng_key_gen)
    aux = (x, 0, rng_key_gen)
    aux, _ = jax.lax.scan(evolve, aux, jnp.arange(num_temps)[::-1])  # reverse temperatures

    x, w, _ = aux
    return x, w, None


def per_sample_eubo(seed, params_flat, unflatten, params_fixed, log_prob, target_samples):
    params_train, params_notrain = unflatten(params_flat)
    params_notrain = jax.lax.stop_gradient(params_notrain)
    params = {**params_train, **params_notrain}  # Gets all parameters in single place
    dim, num_temps, apply_fun_approx_network, alg = params_fixed

    if num_temps >= 1:
        gridref_y = jnp.cumsum(params["mgridref_y"]) / jnp.sum(params["mgridref_y"])
        gridref_y = jnp.concatenate([jnp.array([0.0]), gridref_y])
        betas = jnp.interp(params["target_x"], params["gridref_x"], gridref_y)
    else:
        raise ValueError('Number of temperatures smaller 1.')

    rng_key_gen = jax.random.PRNGKey(seed)

    rng_key, rng_key_gen = jax.random.split(rng_key_gen)
    x = target_samples
    w = log_prob(x)

    if num_temps >= 1:
        rng_key, rng_key_gen = jax.random.split(rng_key_gen)
        x, w_t, _ = evolve_overdamped_langevin_reverse(
            x, betas, params, rng_key, params_fixed, log_prob, sample_kernel, log_prob_kernel
        )

        w += w_t

    # Update weight with final model evaluation
    w = w - bd.log_prob(params["bd"], x)
    return -1.0 * w, (x, _)

