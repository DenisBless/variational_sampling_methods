from typing import Tuple, Callable, List, Optional, Union
import abc

import chex
import matplotlib.pyplot as plt

LogProbFn = Callable[[chex.Array], chex.Array]


class Target(abc.ABC):
    """Abstraction of target distribution that allows our training and evaluation scripts to be generic."""

    def __init__(self,
                 dim: int,
                 log_Z: Optional[float],
                 can_sample: bool,
                 ):
        self._dim = dim
        self._log_Z = log_Z
        self._can_sample = can_sample

    @property
    def dim(self) -> int:
        """Dimensionality of the problem."""
        return self._dim

    @property
    def can_sample(self) -> bool:
        """Whether the target may be sampled form."""
        return self._can_sample

    @property
    def log_Z(self) -> Union[float, None]:
        """Log normalizing constant if available."""
        return self._log_Z

    def sample(self, seed: chex.PRNGKey, sample_shape: chex.Shape) -> chex.Array:
        raise NotImplemented

    @abc.abstractmethod
    def log_prob(self, value: chex.Array) -> chex.Array:
        """(Possibly unnormalized) target probability density."""

    @abc.abstractmethod
    def visualise(self, samples: chex.Array, axes: List[plt.Axes] = None, show: bool = False, prefix: str = '') -> dict:
        """Visualise samples from the model."""
