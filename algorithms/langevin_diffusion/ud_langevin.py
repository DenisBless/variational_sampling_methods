import jax
import jax.numpy as jnp
import algorithms.langevin_diffusion.base_dist as bd
from algorithms.langevin_diffusion.ld_utils import sample_kernel, log_prob_kernel


def evolve_underdamped_langevin(
    x,
    betas,
    params,
    rng_key_gen,
    params_fixed,
    log_prob_model,
    sample_kernel,
    log_prob_kernel,
    apply_fun_approx_network=None,
    use_approx_network=False,
):
    def U(z, beta):
        return -1.0 * (
            beta * log_prob_model(z) + (1.0 - beta) * bd.log_prob(params["bd"], z)
        )

    def evolve(aux, i):
        x, rho, w, rng_key_gen = aux
        beta = betas[i]

        # Forward kernel
        eta_aux = params["gamma"] * params["eps"]
        fk_rho_mean = rho * (1.0 - eta_aux)
        scale = jnp.sqrt(2.0 * eta_aux)

        rng_key, rng_key_gen = jax.random.split(rng_key_gen)
        rho_prime = sample_kernel(rng_key, fk_rho_mean, scale)

        # Leapfrog step
        rho_prime_prime = rho_prime - params["eps"] * jax.grad(U)(x, beta) / 2.0
        x_new = x + params["eps"] * rho_prime_prime
        rho_new = rho_prime_prime - params["eps"] * jax.grad(U)(x_new, beta) / 2.0

        # Backwards kernel
        if alg == 'UHA':  # Uncorrected Hamiltonian Annealing (UHA)
            bk_rho_mean = rho_prime * (1.0 - eta_aux)

        elif alg == 'LDVI':  # Langevin Diffusion Variational Inference (LDVI)
            input_approx_network = jnp.concatenate([x, rho_prime])
            bk_rho_mean = (rho_prime * (1.0 - eta_aux)
                           + 2 * eta_aux * apply_fun_approx_network(params["approx_network"], input_approx_network, i))

        else:
            raise ValueError(f'{alg} is not supported.')


        # Evaluate kernels
        fk_log_prob = log_prob_kernel(rho_prime, fk_rho_mean, scale)
        bk_log_prob = log_prob_kernel(rho, bk_rho_mean, scale)

        # Update weight and return
        w += bk_log_prob - fk_log_prob

        rng_key, rng_key_gen = jax.random.split(rng_key_gen)
        aux = x_new, rho_new, w, rng_key_gen
        return aux, None

    dim, num_temps, apply_fun_approx_network, alg = params_fixed
    # Sample initial momentum
    rng_key, rng_key_gen = jax.random.split(rng_key_gen)
    rho = jax.random.normal(rng_key, shape=(x.shape[0],))  # (dim,)

    # Add initial momentum term to w
    w = 0.0
    w = w - log_prob_kernel(rho, jnp.zeros(x.shape[0]), 1.0)

    # Evolve system
    rng_key, rng_key_gen = jax.random.split(rng_key_gen)
    aux = (x, rho, w, rng_key_gen)

    aux, _ = jax.lax.scan(evolve, aux, jnp.arange(num_temps))

    x, rho, w, _ = aux

    # Add final momentum term to w
    w = w + log_prob_kernel(rho, jnp.zeros(x.shape[0]), 1.0)

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
        x, w_mom, _ = evolve_underdamped_langevin(
            x, betas, params, rng_key, params_fixed, log_prob, sample_kernel, log_prob_kernel
        )

        w += w_mom

    # Update weight with final model evaluation
    w = w + log_prob(x)
    return -1.0 * w, (x, _)


def compute_elbo(seeds, params_flat, unflatten, params_fixed, log_prob):
    elbos, (x, _) = jax.vmap(per_sample_elbo, in_axes=(0, None, None, None, None))(
        seeds, params_flat, unflatten, params_fixed, log_prob
    )
    return elbos.mean(), (elbos, x)


def evolve_underdamped_langevin_reverse(
    x,
    betas,
    params,
    rng_key_gen,
    params_fixed,
    log_prob_model,
    sample_kernel,
    log_prob_kernel,
    apply_fun_approx_network=None,
    use_approx_network=False,
):
    def U(z, beta):
        return -1.0 * (
            beta * log_prob_model(z) + (1.0 - beta) * bd.log_prob(params["bd"], z)
        )

    def evolve(aux, i):  # todo: currently not working correctly
        x, rho, w, rng_key_gen = aux
        beta = betas[i]

        eta_aux = params["gamma"] * params["eps"]
        scale = jnp.sqrt(2.0 * eta_aux)

        # Backwards kernel
        if alg == 'UHA':  # Uncorrected Hamiltonian Annealing (UHA)
            bk_rho_mean = rho * (1.0 - eta_aux)

        elif alg == 'LDVI':  # Langevin Diffusion Variational Inference (LDVI)
            input_approx_network = jnp.concatenate([x, rho])
            bk_rho_mean = (rho * (1.0 - eta_aux)
                           + 2 * eta_aux * apply_fun_approx_network(params["approx_network"], input_approx_network, i))

        else:
            raise ValueError(f'{alg} is not supported.')

        # Sample Backward kernel
        rng_key, rng_key_gen = jax.random.split(rng_key_gen)
        rho_prime = sample_kernel(rng_key, bk_rho_mean, scale)

        # Leapfrog step
        rho_prime_prime = rho_prime + params["eps"] * jax.grad(U)(x, beta) / 2.0
        x_new = x - params["eps"] * rho_prime_prime
        rho_new = rho_prime_prime + params["eps"] * jax.grad(U)(x_new, beta) / 2.0

        # Forward kernel
        fk_rho_mean = rho_prime * (1.0 - eta_aux)

        # Evaluate kernels
        fk_log_prob = log_prob_kernel(rho, fk_rho_mean, scale)
        bk_log_prob = log_prob_kernel(rho_prime, bk_rho_mean, scale)

        # Update weight and return
        w += bk_log_prob - fk_log_prob

        rng_key, rng_key_gen = jax.random.split(rng_key_gen)
        aux = x_new, rho_new, w, rng_key_gen
        return aux, None

    dim, num_temps, apply_fun_approx_network, alg = params_fixed
    # Sample initial momentum
    rng_key, rng_key_gen = jax.random.split(rng_key_gen)
    rho = jax.random.normal(rng_key, shape=(x.shape[0],))  # (dim,)

    # Add initial momentum term to w
    w = 0.0
    w = w + log_prob_kernel(rho, jnp.zeros(x.shape[0]), 1.0)

    # Evolve system
    rng_key, rng_key_gen = jax.random.split(rng_key_gen)
    aux = (x, rho, w, rng_key_gen)

    aux, _ = jax.lax.scan(evolve, aux, jnp.arange(num_temps)[::-1])  # reverse temperatures

    x, rho, w, _ = aux

    # Add final momentum term to w
    w = w - log_prob_kernel(rho, jnp.zeros(x.shape[0]), 1.0)

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
        x, w_t, _ = evolve_underdamped_langevin_reverse(
            x, betas, params, rng_key, params_fixed, log_prob, sample_kernel, log_prob_kernel
        )

        w += w_t

    # Update weight with final model evaluation
    w = w - bd.log_prob(params["bd"], x)
    return -1.0 * w, (x, _)
