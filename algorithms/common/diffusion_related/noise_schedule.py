import jax.numpy as jnp


def get_linear_noise_schedule(total_steps, sigma_min=0.01, sigma_max=10., reverse=True):
    if reverse:
        def linear_noise_schedule(step):
            t = step / total_steps
            return 0.5 * ((1 - t) * sigma_min + t * sigma_max)
    else:
        def linear_noise_schedule(step):
            t = (total_steps - step) / total_steps
            return 0.5 * ((1 - t) * sigma_min + t * sigma_max)

    return linear_noise_schedule


def get_cosine_noise_schedule(total_steps, sigma_min=0.01, sigma_max=10., s=0.008, pow=2, reverse=True):
    if reverse:
        def cosine_noise_schedule(step):
            t = step / total_steps
            offset = 1 + s
            return 0.5 * (sigma_max - sigma_min) * jnp.cos(0.5 * jnp.pi * (offset - t) / offset) ** pow + 0.5 * sigma_min
    else:
        def cosine_noise_schedule(step):
            t = (total_steps - step) / total_steps
            offset = 1 + s
            return 0.5 * (sigma_max - sigma_min) * jnp.cos(0.5 * jnp.pi * (offset - t) / offset) ** pow + 0.5 * sigma_min

    return cosine_noise_schedule


def get_constant_noise_schedule(value, reverse=True):
    def constant_noise_schedule(step):
        return jnp.array(value)

    return constant_noise_schedule
