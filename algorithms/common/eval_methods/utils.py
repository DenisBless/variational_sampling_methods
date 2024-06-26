import jax.numpy as jnp

from utils.path_utils import project_path


def moving_averages(dictionary, window_size=5):
    mov_avgs = {}
    for key, value in dictionary.items():
        try:
            if not 'mov_avg' in key:
                mov_avgs[f'{key}_mov_avg'] = [jnp.mean(jnp.array(value[-min(len(value), window_size):]), axis=0)]
        except:
            pass
    return mov_avgs


def extract_last_entry(dictionary):
    last_entries = {}
    for key, value in dictionary.items():
        try:
            last_entries[key] = value[-min(len(value), 1)]
        except:
            pass
    return last_entries


def save_samples(cfg, logger, samples):
    if len(logger['KL/elbo']) > 1:
        if logger['KL/elbo'][-1] >= jnp.max(jnp.array(logger['KL/elbo'][:-1])):
            jnp.save(project_path(f'{cfg.log_dir}/samples_{cfg.algorithm.name}_{cfg.target.name}_{cfg.target.dim}D_seed{cfg.seed}'), samples)
        else:
            return
    else:
        jnp.save(project_path(f'{cfg.log_dir}/samples_{cfg.algorithm.name}_{cfg.target.name}_{cfg.target.dim}D_seed{cfg.seed}'),
                 samples)


def compute_reverse_ess(log_weights, eval_samples):
    # Subtract the maximum log weight for numerical stability
    max_log_weight = jnp.max(log_weights)
    stable_log_weights = log_weights - max_log_weight

    # Compute the importance weights in a numerically stable way
    is_weights = jnp.exp(stable_log_weights)

    # Compute the sums needed for ESS
    sum_is_weights = jnp.sum(is_weights)
    sum_is_weights_squared = jnp.sum(is_weights ** 2)

    # Calculate the effective sample size (ESS)
    ess = (sum_is_weights ** 2) / (eval_samples * sum_is_weights_squared)

    return ess


if __name__ == '__main__':
    # Example dictionary
    example_dict = {
        'key1': [1, 2, 3, 4],
        'key2': [5, 6],
        'key3': []
    }

    # Convert the dictionary values to JAX arrays
    jax_example_dict = {key: jnp.array(value) for key, value in example_dict.items()}

    # Compute moving average over the last five entries
    result_dict = extract_last_entry(jax_example_dict)

    print(result_dict)
