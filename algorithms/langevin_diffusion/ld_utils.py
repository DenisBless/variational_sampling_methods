import functools
import pickle

import jax
import numpyro.distributions as npdist

from utils.path_utils import project_path


def sample_kernel(rng_key, mean, scale):
    eps = jax.random.normal(rng_key, shape=(mean.shape[0],))
    return mean + scale * eps


def log_prob_kernel(x, mean, scale):
    dist = npdist.Independent(npdist.Normal(loc=mean, scale=scale), 1)
    return dist.log_prob(x)


@functools.partial(jax.jit, static_argnums=(1, 2))
def collect_eps(params_flat, unflatten, trainable):
    if "eps" in trainable:
        return unflatten(params_flat)[0]["eps"]
    else:
        return unflatten(params_flat)[1]["eps"]


@functools.partial(jax.jit, static_argnums=(1, 2))
def collect_gamma(params_flat, unflatten, trainable):
    if "gamma" in trainable:
        return unflatten(params_flat)[0]["gamma"]
    return 0.0


def save_model(model_path, params_flat, unflatten, config, step):
    params_train, params_notrain = unflatten(params_flat)
    pickle.dump(params_train, open(project_path(f'{model_path}/{step}_train.pkl'), 'wb'))
    pickle.dump(params_notrain, open(project_path(f'{model_path}/{step}_notrain.pkl'), 'wb'))


def load_model(file_path, config, step):
    return pickle.load(open(file_path, 'rb'))
