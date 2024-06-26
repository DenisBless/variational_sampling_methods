import distrax
import jax.random

from algorithms.scld.prioritised_buffer import build_prioritised_buffer
from targets.funnel import Funnel
import jax.numpy as jnp

if __name__ == '__main__':
    DIM = 2
    SIGMA = 5
    BATCHSIZE = 500
    SAMPLESIZE = 5000
    target = Funnel(dim=2)

    def geom_avg(x, t):
        return t * target.log_prob(x) + (1-t) * distrax.MultivariateNormalDiag(jnp.zeros(DIM), jnp.ones(DIM) * 1).log_prob(x)

    buffer = build_prioritised_buffer(DIM, 0,
                                      jnp.array(1e6, dtype=int),
                                      min_length_to_sample=jnp.array(BATCHSIZE, dtype=int),
                                      sample_with_replacement=False,
                                      prioritized=True)

    sampler = distrax.MultivariateNormalDiag(jnp.zeros(DIM), jnp.ones(DIM) * SIGMA)
    samples = sampler.sample(seed=jax.random.PRNGKey(0), sample_shape=(1, SAMPLESIZE, ))
    # uniform_weights = jnp.ones([1, SAMPLESIZE])
    # buffer_state = buffer.init(samples, uniform_weights)
    target.visualise(samples[0], show=True)  # Samples in buffer
    for temp in [0, 0.1, 0.5, 0.8, 1]:
        target_weights = geom_avg(samples[0], temp)
        buffer_state = buffer.init(samples, target_weights.reshape(1, -1))
        buffer_samples = buffer.sample(jax.random.PRNGKey(1), buffer_state, BATCHSIZE)
        target.visualise(buffer_samples[-1], show=True)

