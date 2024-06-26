import jax
import jax.numpy as jnp
from jax.lax import stop_gradient
import algorithms.langevin_diffusion.base_dist as bd
from algorithms.langevin_diffusion.ld_utils import sample_kernel, log_prob_kernel


def evolve_overdamped_cmcd(
        z,
        betas,
        params,
        rng_key_gen,
        params_fixed,
        log_prob_model,
        sample_kernel,
        log_prob_kernel,
        grad_clipping=False,
        eps_schedule='None'
):
    def U(z, beta):
        return -1.0 * (
                beta * log_prob_model(z) + (1.0 - beta) * bd.log_prob(params["bd"], z)
        )

    def gradU(z, beta, clip=1e3):
        p = lambda z: bd.log_prob(params["bd"], z)
        gp = jax.grad(p)(z)
        u = lambda z: log_prob_model(z)
        gu = jax.grad(u)(z)
        guc = jnp.clip(gu, -clip, clip)
        return -1.0 * (beta * guc + (1.0 - beta) * gp)

    dim, num_temps, apply_fun_approx_network, alg = params_fixed

    def _linear_eps_schedule(init_eps, i, final_eps=0.0001):
        # Implement linear decay b/w init_eps and final_eps
        return (final_eps - init_eps) / (num_temps - 1) * i + init_eps

    def _cosine_eps_schedule(init_eps, i, s=0.008):
        # Implement cosine decay b/w init_eps and final_eps
        phase = i / num_temps

        decay = jnp.cos((phase + s) / (1 + s) * 0.5 * jnp.pi) ** 2

        return init_eps * decay

    def evolve(aux, i, stable=grad_clipping):
        z, w, rng_key_gen = aux

        beta = betas[i]

        uf = gradU(z, beta) if stable else jax.grad(U)(z, beta)

        if eps_schedule == "linear":
            eps = _linear_eps_schedule(params["eps"], i)
        elif eps_schedule == "cossq":
            eps = _cosine_eps_schedule(params["eps"], i)
        else:
            eps = params["eps"]

        fk_mean = z - eps * uf - eps * apply_fun_approx_network(params["approx_network"], z, i)

        scale = jnp.sqrt(2 * eps)

        # Sample
        rng_key, rng_key_gen = jax.random.split(rng_key_gen)
        z_new = sample_kernel(rng_key, fk_mean, scale)

        # Backwards kernel
        # ub = jax.grad(U)(z_new, beta)
        ub = gradU(z_new, beta) if stable else jax.grad(U)(z_new, beta)

        bk_mean = (z_new - eps * ub + eps * apply_fun_approx_network(params["approx_network"], z_new, i + 1))

        # Evaluate kernels
        fk_log_prob = log_prob_kernel(z_new, fk_mean, scale)
        bk_log_prob = log_prob_kernel(z, bk_mean, scale)

        # Update weight and return
        w += bk_log_prob - fk_log_prob
        rng_key, rng_key_gen = jax.random.split(rng_key_gen)
        aux = z_new, w, rng_key_gen
        return aux, None

    # Evolve system
    rng_key, rng_key_gen = jax.random.split(rng_key_gen)
    aux = (z, 0, rng_key_gen)
    aux, _ = jax.lax.scan(evolve, aux, jnp.arange(num_temps))

    z, w, _ = aux
    return z, w, None


def per_sample_elbo(seed, params_flat, unflatten, params_fixed, log_prob, stop_grad=False,
                    grad_clipping=False,
                    eps_schedule="none", ):
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

    if stop_grad:
        jax.lax.stop_gradient(x)
    w = -bd.log_prob(params["bd"], x)

    if num_temps >= 1:
        rng_key, rng_key_gen = jax.random.split(rng_key_gen)
        x, w_t, _ = evolve_overdamped_cmcd(
            x, betas, params, rng_key, params_fixed, log_prob, sample_kernel, log_prob_kernel, grad_clipping, eps_schedule
        )

        w += w_t

    # Update weight with final model evaluation
    w = w + log_prob(x)
    return -1.0 * w, (x, _)


def compute_elbo(seeds, params_flat, unflatten, params_fixed, log_prob,
                 grad_clipping=False,
                 eps_schedule="none", ):
    elbos, (x, _) = jax.vmap(per_sample_elbo, in_axes=(0, None, None, None, None, None, None, None))(
        seeds, params_flat, unflatten, params_fixed, log_prob, False, grad_clipping, eps_schedule
    )
    return elbos.mean(), (elbos, x)


def compute_log_var(seeds, params_flat, unflatten, params_fixed, log_prob,
                    grad_clipping=False,
                    eps_schedule="none", ):
    elbos, (x, _) = jax.vmap(per_sample_elbo, in_axes=(0, None, None, None, None, None, None, None))(
        seeds, params_flat, unflatten, params_fixed, log_prob, True, grad_clipping, eps_schedule
    )
    return jnp.clip(elbos.var(ddof=0), -1e7, 1e7), (elbos, x)


def evolve_overdamped_cmcd_reverse(
        z,
        betas,
        params,
        rng_key_gen,
        params_fixed,
        log_prob_model,
        sample_kernel,
        log_prob_kernel,
        grad_clipping=False,
        eps_schedule='None',
):
    def U(z, beta):
        return -1.0 * (
                beta * log_prob_model(z) + (1.0 - beta) * bd.log_prob(params["bd"], z)
        )

    def gradU(z, beta, clip=1e3):
        p = lambda z: bd.log_prob(params["bd"], z)
        gp = jax.grad(p)(z)
        u = lambda z: log_prob_model(z)
        gu = jax.grad(u)(z)
        guc = jnp.clip(gu, -clip, clip)
        return -1.0 * (beta * guc + (1.0 - beta) * gp)

    dim, num_temps, apply_fun_approx_network, alg = params_fixed

    def _linear_eps_schedule(init_eps, i, final_eps=0.0001):
        # Implement linear decay b/w init_eps and final_eps
        return (final_eps - init_eps) / (num_temps - 1) * i + init_eps

    def _cosine_eps_schedule(init_eps, i, s=0.008):
        # Implement cosine decay b/w init_eps and final_eps
        phase = i / num_temps

        decay = jnp.cos((phase + s) / (1 + s) * 0.5 * jnp.pi) ** 2

        return init_eps * decay

    def evolve(aux, i, stable=grad_clipping):
        z, w, rng_key_gen = aux
        beta = betas[i]

        if eps_schedule == "linear":
            eps = _linear_eps_schedule(params["eps"], i)
        elif eps_schedule == "cossq":
            eps = _cosine_eps_schedule(params["eps"], i)
        else:
            eps = params["eps"]

        scale = jnp.sqrt(2 * eps)

        # Backwards kernel
        ub = gradU(z, beta) if stable else jax.grad(U)(z, beta)
        bk_mean = (z - eps * ub + eps * apply_fun_approx_network(params["approx_network"], z, i + 1))

        # Sample
        rng_key, rng_key_gen = jax.random.split(rng_key_gen)
        z_new = sample_kernel(rng_key, bk_mean, scale)

        uf = gradU(z_new, beta) if stable else jax.grad(U)(z_new, beta)
        fk_mean = z_new - eps * uf - eps * apply_fun_approx_network(params["approx_network"], z_new, i)

        # Evaluate kernels
        fk_log_prob = log_prob_kernel(z, fk_mean, scale)
        bk_log_prob = log_prob_kernel(z_new, bk_mean, scale)

        # Update weight and return
        w += bk_log_prob - fk_log_prob
        rng_key, rng_key_gen = jax.random.split(rng_key_gen)
        aux = z_new, w, rng_key_gen
        return aux, None

    # Evolve system
    rng_key, rng_key_gen = jax.random.split(rng_key_gen)
    aux = (z, 0, rng_key_gen)
    aux, _ = jax.lax.scan(evolve, aux, jnp.arange(num_temps)[::-1])

    z, w, _ = aux
    return z, w, None


def per_sample_eubo(seed, params_flat, unflatten, params_fixed, log_prob, target_samples,
                    grad_clipping=False,
                    eps_schedule="none", ):
    params_train, params_notrain = unflatten(params_flat)
    params_notrain = jax.lax.stop_gradient(params_notrain)
    params = {**params_train, **params_notrain}  # Gets all parameters in single place
    dim, num_temps, apply_fun_sn, alg = params_fixed

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
        x, w_t, _ = evolve_overdamped_cmcd_reverse(
            x, betas, params, rng_key, params_fixed, log_prob, sample_kernel, log_prob_kernel,
            grad_clipping=grad_clipping,
            eps_schedule=eps_schedule,
        )

        w += w_t

    # Update weight with final model evaluation
    w = w - bd.log_prob(params["bd"], x)
    return -1.0 * w, (x, _)
