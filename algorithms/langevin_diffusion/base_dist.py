import jax
import jax.numpy as np
import numpyro.distributions as npdist


def encode_params(mean, logdiag):
    return {"mean": mean, "logdiag": logdiag}


def decode_params(params):
    mean, logdiag = params["mean"], params["logdiag"]
    return mean, logdiag


def to_scale(logdiag):
    return np.exp(logdiag)


def _initialize(dim, init_sigma=1.0):
    mean = np.zeros(dim)
    logdiag = np.ones(dim) * np.log(init_sigma)
    return encode_params(mean, logdiag)


def build(params):
    mean, logdiag = decode_params(params)
    return npdist.Independent(npdist.Normal(loc=mean, scale=to_scale(logdiag)), 1)


def _log_prob(z, params):
    dist = build(params)
    return dist.log_prob(z)


def log_prob_frozen(z, params):
    dist = build(jax.lax.stop_gradient(params))
    return dist.log_prob(z)


def entropy(params):
    dist = build(params)
    return dist.entropy()


def reparameterize(params, eps):
    mean, logdiag = decode_params(params)
    return to_scale(logdiag) * eps + mean


def sample_eps(rng_key, dim):
    return jax.random.normal(rng_key, shape=(dim,))


def _sample_rep(rng_key, params):
    mean, _ = decode_params(params)
    dim = mean.shape[0]
    eps = sample_eps(rng_key, dim)
    return reparameterize(params, eps)


def initialize(dim, init_sigma=1.0):
    return _initialize(dim, init_sigma=init_sigma)


def sample_rep(rng_key, vdparams):
    return _sample_rep(rng_key, vdparams)


def log_prob(vdparams, z):
    return _log_prob(z, vdparams)
