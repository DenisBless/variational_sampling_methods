from typing import NamedTuple, Callable
import chex
import jax.numpy as jnp
import jax


class SampleDBState(NamedTuple):
    samples: chex.Array
    means: chex.Array
    chols: chex.Array
    inv_chols: chex.Array
    target_lnpdfs: chex.Array
    target_grads: chex.Array
    mapping: chex.Array
    num_samples_written: chex.Array


class SampleDB(NamedTuple):
    init_sampleDB_state: Callable
    add_samples: Callable
    get_random_sample: Callable
    get_newest_samples: Callable
    update_num_samples_written: Callable


def setup_sampledb(DIM, KEEP_SAMPLES, MAX_SAMPLES, DIAGONAL_COVS, DESIRED_SAMPLES_PER_COMPONENT) -> SampleDB:
    def init_sample_db_state():
        if DIAGONAL_COVS:
            chols = jnp.zeros((0, DIM))
            inv_chols = jnp.zeros((0, DIM))
        else:
            chols = jnp.zeros((0, DIM, DIM))
            inv_chols = jnp.zeros((0, DIM, DIM))

        return SampleDBState(samples=jnp.zeros((0, DIM)),
                             means=jnp.zeros((0, DIM)),
                             chols=chols,
                             inv_chols=inv_chols,
                             target_lnpdfs=jnp.zeros(0),
                             target_grads=jnp.zeros((0, DIM)),
                             mapping=jnp.zeros(0, dtype=jnp.int32),
                             num_samples_written=jnp.zeros((1,), dtype=jnp.int32),
                             )

    def add_samples(sampledb_state: SampleDBState, new_samples, new_means, new_chols, new_target_lnpdfs,
                    new_target_grads, new_mapping):
        def _remove_every_nth_sample(sampledb_state: SampleDBState, N: int):
            used_indices, reduced_mapping = jnp.unique(sampledb_state.mapping[::N])

            return SampleDBState(num_samples_written=sampledb_state.num_samples_written,
                                 samples=sampledb_state.samples[::N],
                                 target_lnpdfs=sampledb_state.target_lnpdfs[::N],
                                 target_grads=sampledb_state.target_grads[::N],
                                 mapping=reduced_mapping,
                                 means=sampledb_state.means[used_indices],
                                 chols=sampledb_state.chols[used_indices],
                                 inv_chols=sampledb_state.inv_chols[used_indices],
                                 )

        if MAX_SAMPLES is not None and jnp.shape(new_samples)[0] + jnp.shape(sampledb_state.samples)[0] > MAX_SAMPLES:
            sampledb_state = _remove_every_nth_sample(sampledb_state, 2)

        num_samples_written = sampledb_state.num_samples_written + jnp.shape(new_samples)[0]
        if KEEP_SAMPLES:
            mapping = jnp.concatenate((sampledb_state.mapping, new_mapping + jnp.shape(sampledb_state.chols)[0]))
            means = jnp.concatenate((sampledb_state.means, new_means))
            chols = jnp.concatenate((sampledb_state.chols, new_chols))
            samples = jnp.concatenate((sampledb_state.samples, new_samples))
            target_lnpdfs = jnp.concatenate((sampledb_state.target_lnpdfs, new_target_lnpdfs))
            target_grads = jnp.concatenate((sampledb_state.target_grads, new_target_grads))

            if DIAGONAL_COVS:
                inv_chols = jnp.concatenate((sampledb_state.inv_chols, 1. / new_chols))
            else:
                inv_chols = jnp.concatenate((sampledb_state.inv_chols, jnp.linalg.inv(new_chols)))
        else:
            mapping = new_mapping
            means = new_means
            chols = new_chols
            if DIAGONAL_COVS:
                inv_chols = 1. / new_chols
            else:
                inv_chols = jnp.linalg.inv(new_chols)
            samples = new_samples
            target_lnpdfs = new_target_lnpdfs
            target_grads = new_target_grads

        return SampleDBState(num_samples_written=num_samples_written,
                             samples=samples,
                             target_lnpdfs=target_lnpdfs,
                             target_grads=target_grads,
                             mapping=mapping,
                             means=means,
                             chols=chols,
                             inv_chols=inv_chols,
                             )

    def get_random_sample(sample_db_state: SampleDBState, N: int, seed: chex.PRNGKey):
        chosen_indices = jax.random.permutation(seed, jnp.arange(jnp.shape(sample_db_state.samples)[0]),
                                                independent=True)[:N]
        # chosen_indices = Randomness.get_next_random()
        return sample_db_state.samples[chosen_indices], sample_db_state.target_lnpdfs[chosen_indices]

    def _gaussian_log_pdf(mean, chol, inv_chol, x):
        if DIAGONAL_COVS:
            constant_part = - 0.5 * DIM * jnp.log(2 * jnp.pi) - jnp.sum(jnp.log(chol))
            return constant_part - 0.5 * jnp.sum(jnp.square(jnp.expand_dims(inv_chol, 1)
                                                            * jnp.transpose(jnp.expand_dims(mean, 0) - x)), axis=0)
        else:
            constant_part = - 0.5 * DIM * jnp.log(2 * jnp.pi) - jnp.sum(jnp.log(jnp.diag(chol)))
            return constant_part - 0.5 * jnp.sum(jnp.square(inv_chol @ jnp.transpose(mean - x)), axis=0)

    def get_newest_samples_deprecated(sampledb_state: SampleDBState, N):
        def _compute_log_pdfs(sampledb_state, component_id, sample):
            return jax.lax.cond(component_id == -1,
                                lambda: jnp.full(sample.shape[0], -jnp.inf),
                                lambda: _gaussian_log_pdf(sampledb_state.means[component_id],
                                                          sampledb_state.chols[component_id],
                                                          sampledb_state.inv_chols[component_id], sample))

        if jnp.shape(sampledb_state.samples)[0] == 0 or N == 0:
            return jnp.zeros(0), jnp.zeros((0, DIM)), jnp.zeros(0, dtype=jnp.int32), jnp.zeros(0), jnp.zeros((0, DIM))
        else:
            active_sample_index = jnp.maximum(0, jnp.shape(sampledb_state.samples)[0] - N)
            active_sample = sampledb_state.samples[active_sample_index:]
            active_target_lnpdfs = sampledb_state.target_lnpdfs[active_sample_index:]
            active_target_grads = sampledb_state.target_grads[active_sample_index:]
            active_mapping = sampledb_state.mapping[active_sample_index:]
            @jax.jit
            def compute_background_pdf():
                active_components, count = jnp.unique(active_mapping, return_counts=True, size=sampledb_state.means.shape[0], fill_value=-1)
                weights = count / jnp.sum(count)
                return jax.nn.logsumexp(jax.vmap(_compute_log_pdfs, in_axes=(None, 0, None))(sampledb_state, active_components, active_sample) + jnp.expand_dims(jnp.log(weights), 1), axis=0)
            log_pdfs = compute_background_pdf()

            return log_pdfs, active_sample, active_mapping, active_target_lnpdfs, active_target_grads

    def get_newest_samples(sampledb_state: SampleDBState, N):
        # use other implementation for original behavious, N % DESIRED_SAMPLES_PER_COMPONENT = 0 is needed to ensure uniform weights

        @jax.jit
        def _compute_log_pdfs(sampledb_state, component_id, sample):
            return _gaussian_log_pdf(sampledb_state.means[component_id],
                                     sampledb_state.chols[component_id],
                                     sampledb_state.inv_chols[component_id], sample)

        chex.assert_equal(N % DESIRED_SAMPLES_PER_COMPONENT, 0)
        if jnp.shape(sampledb_state.samples)[0] == 0 or N == 0:
            return jnp.zeros(0), jnp.zeros((0, DIM)), jnp.zeros(0, dtype=jnp.int32), jnp.zeros(0), jnp.zeros((0, DIM))
        else:
            active_sample_index = jnp.maximum(0, jnp.shape(sampledb_state.samples)[0] - N)
            active_sample = sampledb_state.samples[active_sample_index:]
            active_target_lnpdfs = sampledb_state.target_lnpdfs[active_sample_index:]
            active_target_grads = sampledb_state.target_grads[active_sample_index:]
            active_mapping = sampledb_state.mapping[active_sample_index:]
            num_active_comps = N // DESIRED_SAMPLES_PER_COMPONENT
            active_components = jnp.arange(jnp.maximum(sampledb_state.means.shape[0] - num_active_comps, 0),
                                           sampledb_state.means.shape[0])
            weights_test = jnp.ones_like(active_components) / jnp.shape(active_components)[0]

            def compute_background_pdf():
                log_pdfs = jax.vmap(_compute_log_pdfs, in_axes=(None, 0, None))(sampledb_state, active_components,
                                                                                active_sample) + jnp.expand_dims(
                    jnp.log(weights_test), 1)
                return jax.nn.logsumexp(log_pdfs, axis=0)

            log_pdfs = compute_background_pdf()
            return log_pdfs, active_sample, active_mapping, active_target_lnpdfs, active_target_grads


    def update_num_samples_written(sample_db_state: SampleDBState, num_samples_written):
        return SampleDBState(samples=sample_db_state.samples,
                             means=sample_db_state.means,
                             chols=sample_db_state.chols,
                             inv_chols=sample_db_state.inv_chols,
                             target_lnpdfs=sample_db_state.target_lnpdfs,
                             target_grads=sample_db_state.target_grads,
                             mapping=sample_db_state.mapping,
                             num_samples_written=num_samples_written)

    return SampleDB(init_sampleDB_state=init_sample_db_state,
                    add_samples=add_samples,
                    get_random_sample=get_random_sample,
                    get_newest_samples=get_newest_samples,
                    update_num_samples_written=update_num_samples_written
                    )
