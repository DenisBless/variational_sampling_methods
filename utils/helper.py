import gc
import jax
import jax.numpy as jnp


def flatten_dict(d, parent_key='', sep='_'):
    """
    Flatten a nested dictionary into a flat dictionary.

    Args:
        d (dict): The dictionary to flatten.
        parent_key (str): The parent key for the current level of the dictionary.
        sep (str): The separator to use between keys.

    Returns:
        dict: The flattened dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def reset_device_memory(delete_objs=True):
    """Free all tracked DeviceArray memory and delete objects.
  Args:
    delete_objs: bool: whether to delete all live DeviceValues or just free.
  Returns:
    number of DeviceArrays that were manually freed.
  """
    dvals = (x for x in gc.get_objects() if isinstance(x, jax.xla.DeviceValue))
    n_deleted = 0
    for dv in dvals:
        if not isinstance(dv, jax.xla.DeviceConstant):
            try:
                dv._check_if_deleted()  # pylint: disable=protected-access
                dv.delete()
                n_deleted += 1
            except ValueError:
                pass
        if delete_objs:
            del dv
    del dvals
    gc.collect()

    backend = jax.lib.xla_bridge.get_backend()
    for buf in backend.live_buffers(): buf.delete()
    return n_deleted


def stable_mean(x):
    # Create a mask where `True` indicates non-NaN values
    nan_check = ~jnp.isnan(x)
    inf_check = ~jnp.isinf(x)
    mask = nan_check & inf_check

    # Replace NaNs with zero
    x = jnp.where(mask, x, 0)

    # Compute the sum of non-NaN values
    total_sum = jnp.sum(x)

    # Compute the number of non-NaN values
    count = jnp.sum(mask)

    # Compute the mean of non-NaN values
    mean_value = total_sum / count

    return mean_value


def replace_invalid(x, replacement=0.):
    # Create a mask where `True` indicates non-NaN values
    nan_check = ~jnp.isnan(x)
    inf_check = ~jnp.isinf(x)
    mask = nan_check & inf_check

    # Replace NaNs with zero
    x = jnp.where(mask, x, replacement)

    # Compute the number of non-NaN values
    invalid_count = jnp.sum(mask)

    return x, invalid_count


def inverse_softplus(x):
    return jnp.log(jnp.exp(x) - 1)


if __name__ == '__main__':
    init_std = 10
    a = inverse_softplus(10)
    print(jax.nn.softplus(a))
