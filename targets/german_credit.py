import chex
import numpy as np
import jax.numpy as jnp
import jax
from utils.path_utils import project_path
from targets.base_target import Target


class GermanCredit(Target):
    def __init__(self, log_Z=None, can_sample=False, sample_bounds=None):
        super().__init__(dim=25, log_Z=log_Z, can_sample=can_sample)
        data = np.loadtxt(project_path("targets/data/german.data-numeric"))
        X = data[:, :-1]
        X /= jnp.std(X, 0)[jnp.newaxis, :]
        X = jnp.hstack((jnp.ones((len(X), 1)), X))
        self.data = jnp.array(X, dtype=jnp.float32)
        self.labels = data[:, -1] - 1
        self.num_dimensions = self.data.shape[1]
        self._prior_std_const = jnp.array(10., dtype=jnp.float32)
        self.prior_mean_const = jnp.array(0., dtype=jnp.float32)
        self.labels = jnp.array(jnp.expand_dims(self.labels.astype(jnp.float32), 1))
        self.const_term = jnp.array(0.5 * jnp.log(2. * jnp.pi), dtype=jnp.float32)

    def log_prob(self, x: chex.Array) -> chex.Array:
        def _log_prob(x: chex.Array):
            features = -jnp.matmul(self.data, x.transpose())
            log_likelihood = jnp.sum(jnp.where(self.labels == 1, jax.nn.log_sigmoid(features),
                                               jax.nn.log_sigmoid(features) - features), axis=0)
            log_prior = jnp.sum(-jnp.log(self._prior_std_const) - self.const_term - 0.5 * jnp.square(
                (x - self.prior_mean_const) / self._prior_std_const), axis=1)
            log_posterior = log_likelihood #+ log_prior
            return log_posterior

        batched = x.ndim == 2
        if not batched:
            x = x[None,]

        log_probs = _log_prob(x)

        if not batched:
            log_probs = jnp.squeeze(log_probs, axis=0)

        return log_probs

    def visualise(self, samples: chex.Array = None, axes=None, show=False, prefix='') -> dict:
        return {}

    def sample(self, seed: chex.PRNGKey, sample_shape: chex.Shape) -> chex.Array:
        ground_truth_samples = np.load(project_path('targets/data/german_credit10k.npy'))

        indices = jax.random.choice(seed, ground_truth_samples.shape[0], shape=sample_shape, replace=False)
        # Use the generated indices to select the subset
        return ground_truth_samples[indices]


if __name__ == "__main__":
    germanCredit = GermanCredit()

    key = jax.random.PRNGKey(42)
    samples = jnp.zeros(shape=(1, 25))
    print(samples)
    print(germanCredit.log_prob(samples))
    print(jax.vmap(germanCredit.log_prob)(samples))

    # grad
    log_prob_grad = jax.vmap(jax.grad(germanCredit.log_prob))(samples)
    print(log_prob_grad)

    print((germanCredit.sample(key, (5,))).shape)