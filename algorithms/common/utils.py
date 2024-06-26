import optax
import jax.numpy as jnp
import jax


def get_optimizer(initial_learning_rate: float,
                  boundaries_and_scales):
    """Get an optimizer possibly with learning rate schedule."""
    if boundaries_and_scales is None:
        return optax.adam(initial_learning_rate)
    else:
        schedule_fn = optax.piecewise_constant_schedule(
            initial_learning_rate,
            boundaries_and_scales[0])
        opt = optax.chain(optax.scale_by_adam(),
                          optax.scale_by_schedule(schedule_fn), optax.scale(-1.))
        return opt


def avg_list_entries(list, num):
    assert len(list) >= num
    print(range(0, len(list) - num))
    return [sum(list[i:i + num]) / float(num) for i in range(0, len(list) - num + 1)]


def reverse_transition_params(transition_params):
    flattened_params, tree = jax.tree_util.tree_flatten(transition_params, is_leaf=None)
    reversed_flattened_params = list(map(lambda w: jnp.flip(w, axis=0), flattened_params))
    return jax.tree_util.tree_unflatten(tree, reversed_flattened_params)


def interpolate_values(values, X):
    # Compute the interpolated values
    interpolated_values = [X] + [X + (X / 2 - X) * t for t in values[1:-1]] + [X / 2]
    return interpolated_values
