from typing import List

import inference_gym.using_jax as gym
import chex
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from targets.base_target import Target


class Lorenz(Target):
    def __init__(self, dim=90, log_Z=None, can_sample=False, sample_bounds=None) -> None:
        super().__init__(dim=dim, log_Z=log_Z, can_sample=can_sample)
        self.data_ndim = dim

        target = gym.targets.ConvectionLorenzBridge()
        self.target = gym.targets.VectorModel(target, flatten_sample_transformations=True)

    def get_dim(self):
        return self.dim

    def log_prob(self, z: chex.Array):
        batched = z.ndim == 2

        if not batched:
            z = z[None,]

        x = self.target.default_event_space_bijector(z)
        log_prob = (self.target.unnormalized_log_prob(x) +
                    self.target.default_event_space_bijector.forward_log_det_jacobian(z, event_ndims=1))

        if not batched:
            log_prob = jnp.squeeze(log_prob, axis=0)

        return log_prob

    def visualise(self, samples: chex.Array = None, axes=None, show=False, prefix='') -> dict:
        return {}

    def sample(self, seed: chex.PRNGKey, sample_shape: chex.Shape) -> chex.Array:
        return None

if __name__ == "__main__":
    key = jax.random.PRNGKey(42)

    lorenz = Lorenz()
    samples = jax.random.normal(key, shape=(10, 90))
    print(lorenz.log_prob(samples))
    log_prob_grad = jax.vmap(jax.grad(lorenz.log_prob))(samples)
    print(log_prob_grad)
