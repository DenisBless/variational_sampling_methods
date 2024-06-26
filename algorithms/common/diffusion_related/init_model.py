import optax
from flax.training import train_state

from algorithms.common.models.pisgrad_net import PISGRADNet
import jax
import jax.numpy as jnp

from algorithms.scld.scld_utils import flattened_traversal


def init_model(key, dim, alg_cfg):
    # Define the model
    model = PISGRADNet(**alg_cfg.model)
    # model = LangevinNetwork(**alg_cfg.model)
    key, key_gen = jax.random.split(key)
    params = model.init(key, jnp.ones([alg_cfg.batch_size, dim]),
                        jnp.ones([alg_cfg.batch_size, 1]),
                        jnp.ones([alg_cfg.batch_size, dim]))

    if alg_cfg.name == 'gfn':
        additional_params = {'logZ': jnp.array((alg_cfg.init_logZ,))}
        params['params'] = {**params['params'], **additional_params}

        optimizer = optax.chain(
            optax.zero_nans(),
            optax.clip(alg_cfg.grad_clip) if alg_cfg.grad_clip > 0 else optax.identity(),
            optax.masked(optax.adam(learning_rate=alg_cfg.step_size),
                         mask=flattened_traversal(lambda path, _: path[-1] != 'logZ')),
            optax.masked(optax.adam(learning_rate=alg_cfg.logZ_step_size),
                         mask=flattened_traversal(lambda path, _: path[-1] == 'logZ')),
        )

    else:
        optimizer = optax.chain(optax.zero_nans(),
                                optax.clip(alg_cfg.grad_clip) if alg_cfg.grad_clip > 0 else optax.identity(),
                                optax.adam(learning_rate=alg_cfg.step_size))

    model_state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

    return model_state
