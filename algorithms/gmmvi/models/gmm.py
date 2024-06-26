import chex
import jax.numpy as jnp
import jax
from typing import NamedTuple, Callable


class GMMState(NamedTuple):
    log_weights: chex.Array
    means: chex.Array
    chol_covs: chex.Array
    num_components: int


class GMM(NamedTuple):
    init_gmm_state: Callable
    sample: Callable
    sample_from_components_no_shuffle: Callable
    add_component: Callable
    remove_component: Callable
    replace_components: Callable
    average_entropy: Callable
    replace_weights: Callable
    component_log_densities: Callable
    log_densities_also_individual: Callable
    log_density: Callable
    log_density_and_grad: Callable


def _setup_initial_mixture_params(NUM_DIM, key, diagonal_covs, num_initial_components, prior_mean, prior_scale,
                                  initial_cov=None):

    if jnp.isscalar(prior_mean):
        prior_mean = prior_mean * jnp.ones(NUM_DIM)

    if jnp.isscalar(prior_scale):
        prior_scale = prior_scale * jnp.ones(NUM_DIM)
    prior = jnp.array(prior_scale) ** 2

    weights = jnp.ones(num_initial_components, dtype=jnp.float32) / num_initial_components
    means = jnp.zeros((num_initial_components, NUM_DIM), dtype=jnp.float32)

    if diagonal_covs:
        if initial_cov is None:
            initial_cov = prior  # use the same initial covariance that was used for sampling the mean
        else:
            initial_cov = initial_cov * jnp.ones(NUM_DIM)

        covs = jnp.full((num_initial_components, NUM_DIM), initial_cov, dtype=jnp.float32)
        for i in range(0, num_initial_components):
            key, subkey = jax.random.split(key)
            if num_initial_components == 1:
                means = means.at[i].set(prior_mean)
            else:
                rand_samples = jax.random.normal(subkey, (NUM_DIM,))
                means = means.at[i].set(prior_mean + jnp.sqrt(prior) * rand_samples)

    else:
        prior = jnp.diag(prior)
        if initial_cov is None:
            initial_cov = prior  # use the same initial covariance that was used for sampling the mean
        else:
            initial_cov = initial_cov * jnp.eye(NUM_DIM)

        covs = jnp.full((num_initial_components, NUM_DIM, NUM_DIM), initial_cov, dtype=jnp.float32)
        for i in range(0, num_initial_components):
            key, subkey = jax.random.split(key)
            if num_initial_components == 1:
                means = means.at[i].set(prior_mean)
            else:
                rand_samples = jax.random.normal(subkey, (NUM_DIM,))
                means = means.at[i].set(prior_mean + jnp.linalg.cholesky(prior) @ rand_samples)

    if diagonal_covs:
        chol_covs = jnp.stack([jnp.sqrt(cov) for cov in covs])
    else:
        chol_covs = jnp.stack([jnp.linalg.cholesky(cov) for cov in covs])

    return weights, means, chol_covs


def setup_sample_fn(sample_from_component_fn: Callable):
    def sample(gmm_state: GMMState, seed: chex.PRNGKey, num_samples: int) -> [chex.Array, chex.Array]:
        weights = jnp.exp(gmm_state.log_weights)
        sampled_components = jax.random.choice(seed, gmm_state.num_components, shape=(num_samples,), p=weights)
        component_count = jnp.bincount(sampled_components, minlength=gmm_state.num_components)

        samples = []
        for i in range(gmm_state.num_components):
            if component_count[i] == 0:
                continue
            seed_i = jax.random.fold_in(seed, i)
            samples.append(sample_from_component_fn(gmm_state, i, component_count[i], seed_i))

        samples = jnp.vstack(samples)
        return jnp.squeeze(samples), sampled_components

    return sample


def setup_sample_from_components_no_shuffle_fn_deprecated(sample_from_component_fn: Callable):

    def sample_from_components_no_shuffle(gmm_state: GMMState, samples_per_component: chex.Array,
                                          seed: chex.PRNGKey) -> [chex.Array, chex.Array]:
        mapping = jnp.repeat(jnp.arange(gmm_state.num_components), samples_per_component)
        samples = []

        for i in range(gmm_state.num_components):
            if samples_per_component[i] == 0:
                continue
            seed_i = jax.random.fold_in(seed, i)
            samples.append(sample_from_component_fn(gmm_state, i, samples_per_component[i], seed_i))

        samples = jnp.vstack(samples)
        return samples, mapping

    return sample_from_components_no_shuffle


def setup_sample_from_components_no_shuffle_fn(sample_from_component_fn: Callable):

    def sample_from_components_no_shuffle(gmm_state: GMMState, DESIRED_SAMPLES, num_components,
                                          seed: chex.PRNGKey) -> [chex.Array, chex.Array]:
        mapping = jnp.repeat(jnp.arange(num_components), DESIRED_SAMPLES)
        samples = jax.vmap(sample_from_component_fn, in_axes=(None, 0, None, 0))(gmm_state,
                                                                                 jnp.arange(num_components),
                                                                                 DESIRED_SAMPLES,
                                                                                 jax.random.split(seed, (num_components,)))

        return jnp.vstack(samples), mapping

    return sample_from_components_no_shuffle


def _normalize_weights(new_log_weights: chex.Array):
    return new_log_weights - jax.nn.logsumexp(new_log_weights)


def replace_weights(gmm_state: GMMState, new_log_weights: chex.Array):
    return GMMState(log_weights=_normalize_weights(new_log_weights),
                    means=gmm_state.means,
                    chol_covs=gmm_state.chol_covs,
                    num_components=gmm_state.num_components)


def remove_component(gmm_state: GMMState, idx):
    return GMMState(log_weights=_normalize_weights(jnp.concatenate((gmm_state.log_weights[:idx],
                                                   gmm_state.log_weights[idx + 1:]), axis=0)),
                    means=jnp.concatenate((gmm_state.means[:idx], gmm_state.means[idx + 1:]), axis=0),
                    chol_covs=jnp.concatenate((gmm_state.chol_covs[:idx], gmm_state.chol_covs[idx + 1:]), axis=0),
                    num_components=gmm_state.num_components - 1)


def replace_components(gmm_state: GMMState, new_means: chex.Array, new_chols: chex.Array) -> GMMState:
    new_means = jnp.stack(new_means, axis=0)
    new_chols = jnp.stack(new_chols, axis=0)
    return GMMState(log_weights=gmm_state.log_weights,
                    means=new_means,
                    chol_covs=new_chols,
                    num_components=gmm_state.num_components)


def setup_get_average_entropy_fn(gaussian_entropy_fn: Callable):
    def get_average_entropy(gmm_state: GMMState) -> jnp.float32:
        gaussian_entropies = jax.vmap(gaussian_entropy_fn)(gmm_state.chol_covs)
        return jnp.sum(jnp.exp(gmm_state.log_weights) * gaussian_entropies)

    return get_average_entropy


def setup_log_density_fn(component_log_densities_fn: Callable):
    def log_density(gmm_state: GMMState, sample: chex.Array) -> chex.Array:
        log_densities = component_log_densities_fn(gmm_state, sample)
        weighted_densities = log_densities + gmm_state.log_weights
        return jax.nn.logsumexp(weighted_densities)

    return log_density


def setup_log_density_and_grad_fn(component_log_densities_fn: Callable):
    def log_density_and_grad(gmm_state: GMMState, sample: chex.Array) -> [chex.Array, chex.Array, chex.Array]:
        def compute_log_densities(sample):
            log_component_dens = component_log_densities_fn(gmm_state, sample)
            log_densities = log_component_dens + gmm_state.log_weights
            x = jax.nn.logsumexp(log_densities, axis=0)
            return x, log_component_dens

        (log_densities, log_component_densities), log_densities_grad = jax.value_and_grad(compute_log_densities,
                                                                                          has_aux=True)(sample)

        return log_densities, log_densities_grad, log_component_densities

    return log_density_and_grad


def setup_log_densities_also_individual_fn(component_log_densities_fn: Callable):
    def log_densities_also_individual(gmm_state: GMMState, sample: chex.Array) -> [chex.Array, chex.Array]:
        component_log_dens = component_log_densities_fn(gmm_state, sample)
        weighted_densities = component_log_dens + gmm_state.log_weights
        return jax.nn.logsumexp(weighted_densities), component_log_dens

    return log_densities_also_individual


def setup_diagonal_gmm(DIM) -> GMM:
    def init_diagonal_gmm_state(seed, num_initial_components, prior_mean, prior_scale, diagonal_covs, initial_cov=None):
        weights, means, chol_covs = _setup_initial_mixture_params(DIM, seed, diagonal_covs, num_initial_components,
                                                                  prior_mean, prior_scale, initial_cov)

        return GMMState(log_weights=_normalize_weights(jnp.log(weights)),
                        means=means,
                        chol_covs=chol_covs,
                        num_components=num_initial_components)

    def sample_from_component(gmm_state: GMMState, index: int, num_samples: int, seed: chex.PRNGKey) -> chex.Array:
        samples = jnp.transpose(jnp.expand_dims(gmm_state.means[index], 1) + jnp.expand_dims(gmm_state.chol_covs[index], 1)
                                * jax.random.normal(seed, (DIM, num_samples)))
        return samples

    def component_log_densities(gmm_state: GMMState, samples: chex.Array) -> chex.Array:
        diffs = jnp.expand_dims(samples, 0) - gmm_state.means
        inv_chol = 1. / gmm_state.chol_covs  # Inverse of diagonal elements
        mahalas = -0.5 * jnp.sum(jnp.square(diffs * inv_chol), axis=-1)
        const_parts = -jnp.sum(jnp.log(gmm_state.chol_covs), axis=1) - 0.5 * DIM * jnp.log(2 * jnp.pi)
        log_pdfs = mahalas + const_parts
        return log_pdfs

    def gaussian_entropy(chol: chex.Array) -> chex.Array:
        return 0.5 * DIM * (jnp.log(2 * jnp.pi) + 1) + jnp.sum(jnp.log(chol))

    def add_component(gmm_state: GMMState, initial_weight: chex.Array, initial_mean: chex.Array,
                      initial_cov: chex.Array):
        return GMMState(log_weights=_normalize_weights(jnp.concatenate((gmm_state.log_weights,
                                                                        jnp.expand_dims(jnp.log(initial_weight),
                                                                                        axis=0)),
                                                                       axis=0)),
                        means=jnp.concatenate((gmm_state.means, jnp.expand_dims(initial_mean, axis=0)), axis=0),
                        chol_covs=jnp.concatenate(
                            (gmm_state.chol_covs, jnp.expand_dims(jnp.sqrt(initial_cov), axis=0)), axis=0),
                        num_components=gmm_state.num_components + 1)

    return GMM(init_gmm_state=init_diagonal_gmm_state,
               sample=setup_sample_fn(sample_from_component),
               sample_from_components_no_shuffle=setup_sample_from_components_no_shuffle_fn(sample_from_component),
               add_component=add_component,
               remove_component=remove_component,
               replace_components=replace_components,
               average_entropy=setup_get_average_entropy_fn(gaussian_entropy),
               replace_weights=replace_weights,
               component_log_densities=component_log_densities,
               log_density=setup_log_density_fn(component_log_densities),
               log_densities_also_individual=setup_log_densities_also_individual_fn(component_log_densities),
               log_density_and_grad=setup_log_density_and_grad_fn(component_log_densities))


def setup_full_cov_gmm(DIM) -> GMM:
    def init_full_cov_gmm_state(seed, num_initial_components, prior_mean, prior_scale, diagonal_covs, initial_cov=None):
        weights, means, chol_covs = _setup_initial_mixture_params(DIM, seed, diagonal_covs, num_initial_components,
                                                                  prior_mean, prior_scale, initial_cov)
        return GMMState(log_weights=_normalize_weights(jnp.log(weights)),
                        means=means,
                        chol_covs=chol_covs,
                        num_components=num_initial_components)

    def sample_from_component(gmm_state: GMMState, index: int, num_samples: int, seed: chex.Array) -> chex.Array:
        return jnp.transpose(jnp.expand_dims(gmm_state.means[index], axis=-1)
                             + gmm_state.chol_covs[index] @ jax.random.normal(key=seed, shape=(DIM, num_samples)))

    def component_log_densities(gmm_state: GMMState, sample: chex.Array) -> chex.Array:
        diffs = jnp.expand_dims(sample, 0) - gmm_state.means
        sqrts = jax.scipy.linalg.solve_triangular(gmm_state.chol_covs, diffs, lower=True)
        mahalas = - 0.5 * jnp.sum(sqrts * sqrts, axis=1)
        const_parts = - 0.5 * jnp.sum(jnp.log(jnp.square(jnp.diagonal(gmm_state.chol_covs, axis1=1, axis2=2))),
                                      axis=1) - 0.5 * DIM * jnp.log(2 * jnp.pi)
        return mahalas + const_parts

    def gaussian_entropy(chol: chex.Array) -> chex.Array:
        return 0.5 * DIM * (jnp.log(2 * jnp.pi) + 1) + jnp.sum(jnp.log(jnp.diag(chol)))

    def add_component(gmm_state: GMMState, initial_weight: chex.Array, initial_mean: chex.Array,
                      initial_cov: chex.Array):
        return GMMState(log_weights=_normalize_weights(jnp.concatenate((gmm_state.log_weights,
                                                                        jnp.expand_dims(jnp.log(initial_weight),
                                                                                        axis=0)),
                                                                       axis=0)),
                        means=jnp.concatenate((gmm_state.means, jnp.expand_dims(initial_mean, axis=0)), axis=0),
                        chol_covs=jnp.concatenate(
                            (gmm_state.chol_covs, jnp.expand_dims(jnp.linalg.cholesky(initial_cov), axis=0)), axis=0),
                        num_components=gmm_state.num_components + 1)

    return GMM(init_gmm_state=init_full_cov_gmm_state,
               sample=setup_sample_fn(sample_from_component),
               sample_from_components_no_shuffle=setup_sample_from_components_no_shuffle_fn(sample_from_component),
               add_component=add_component,
               remove_component=remove_component,
               replace_components=replace_components,
               average_entropy=setup_get_average_entropy_fn(gaussian_entropy),
               replace_weights=replace_weights,
               component_log_densities=component_log_densities,
               log_density=setup_log_density_fn(component_log_densities),
               log_densities_also_individual=setup_log_densities_also_individual_fn(component_log_densities),
               log_density_and_grad=setup_log_density_and_grad_fn(component_log_densities))
