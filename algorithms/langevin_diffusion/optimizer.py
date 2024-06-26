import functools

import jax
import jax.numpy as np
from jax.flatten_util import ravel_pytree


def adam(step_size, b1=0.9, b2=0.999, eps=1e-8):
    # Basically JAX's thing with added projection for some parameters.
    # Assumes ravel_pytree will always work the same way, so no need to update the
    # unflatten function (which may be problematic for jitting stuff)
    def init(x0):
        m0 = np.zeros_like(x0)
        v0 = np.zeros_like(x0)
        return x0, m0, v0

    def update(i, g, state, unflatten, trainable):
        def project(x, unflatten, trainable):
            x_train, x_notrain = unflatten(x)
            if "eps" in trainable:
                x_train["eps"] = np.clip(x_train["eps"], 0.0000001, 0.5)
            if "eta" in trainable:
                x_train["eta"] = np.clip(x_train["eta"], 0, 0.99)
            if "gamma" in trainable:
                x_train["gamma"] = np.clip(x_train["gamma"], 0.001, None)
            if "mgridref_y" in trainable:
                x_train["mgridref_y"] = (
                    jax.nn.relu(x_train["mgridref_y"] - 0.001) + 0.001
                )
            return ravel_pytree((x_train, x_notrain))[0]

        x, m, v = state
        m = (1 - b1) * g + b1 * m  # First moment estimate
        v = (1 - b2) * np.square(g) + b2 * v  # Second moment estimate
        mhat = m / (1 - np.asarray(b1, m.dtype) ** (i + 1))  # Bias correction
        vhat = v / (1 - np.asarray(b2, m.dtype) ** (i + 1))
        x = x - step_size * mhat / (np.sqrt(vhat) + eps)
        x = project(x, unflatten, trainable)
        return x, m, v

    def get_params(state):
        x, _, _ = state
        return x

    return init, update, get_params


