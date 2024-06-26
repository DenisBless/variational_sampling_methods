from typing import List
import inference_gym.using_jax as gym
import chex
import matplotlib.pyplot as plt
from targets.base_target import Target


class Brownian(Target):
    def __init__(self, dim=32, log_Z=None, can_sample=False, sample_bounds=None) -> None:
        super().__init__(dim=dim, log_Z=log_Z, can_sample=can_sample)
        self.data_ndim = dim

        target = gym.targets.BrownianMotionUnknownScalesMissingMiddleObservations()
        self.target = gym.targets.VectorModel(target, flatten_sample_transformations=True)

    def get_dim(self):
        return self.dim

    def log_prob(self, z: chex.Array):
        x = self.target.default_event_space_bijector(z)
        return (self.target.unnormalized_log_prob(
            x) + self.target.default_event_space_bijector.forward_log_det_jacobian(z, event_ndims=1))

    def visualise(self, samples: chex.Array = None, axes: List[plt.Axes] = None, show=False, prefix='') -> dict:
        return {}


    def sample(self, seed: chex.PRNGKey, sample_shape: chex.Shape) -> chex.Array:
        return None
