import functools
import os
import pickle

import haiku as hk
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
import wandb
from absl import app, flags
from ml_collections import config_flags

from targets.nice import NICE
from utils.mode_classifier.data_utils import flatten_nested_dict, setup_training, update_config_dict

config_flags.DEFINE_config_file("config", ("nice_config.py"), "Run configuration.")
FLAGS = flags.FLAGS


def get_config():
    config = ml_collections.ConfigDict()

    config.alpha = 0.05  # 0.000001
    config.batch_size = 1000
    config.hidden_dim = 1000
    config.im_size = 14  # 14
    config.log_interval = 1000
    config.lr = 0.0001
    config.n_bits = 3  # 3
    config.num_epochs = 500
    config.save_interval = 1000
    config.seed = 42
    config.weight_decay = 0.000001

    cwd = os.getcwd()
    # Wandb Configs
    config.wandb = ml_collections.ConfigDict()
    config.wandb.log = True
    config.wandb.project = "cais"
    config.wandb.entity = "shreyaspadhy"
    config.wandb.code_dir = cwd
    config.wandb.name = ""
    config.wandb.log_artifact = True

    return config

def dequantize(x, y, n_bits=3):
    n_bins = 2.0**n_bits
    x = tf.cast(x, tf.float32)
    x = tf.floor(x / 2 ** (8 - n_bits))
    x = x / n_bins
    x = x + tf.random.uniform(x.shape) / n_bins
    return x, y


def resize(x, y, im_size=28):
    """Resize images to desired size."""
    x = tf.image.resize(x, (im_size, im_size))
    return x, y


def logit(x, y, alpha=1e-6):
    """Scales inputs to rance [alpha, 1-alpha] then applies logit transform."""
    x = x * (1 - 2 * alpha) + alpha
    x = tf.math.log(x) - tf.math.log(1.0 - x)
    return x, y


def load_dataset(split: str, batch_size: int, im_size: int, alpha: float, n_bits: int):
    """Loads the dataset as a generator of batches."""
    ds, ds_info = tfds.load("mnist", split=split, as_supervised=True, with_info=True)
    ds = ds.cache()
    ds = ds.map(
        lambda x, y: resize(x, y, im_size=im_size), num_parallel_calls=tf.data.AUTOTUNE
    )
    ds = ds.map(
        lambda x, y: dequantize(x, y, n_bits=n_bits),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.map(
        lambda x, y: logit(x, y, alpha=alpha), num_parallel_calls=tf.data.AUTOTUNE
    )
    ds = ds.shuffle(ds_info.splits["train"].num_examples)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return tfds.as_numpy(ds)


def main(config):
    """Main experiment."""

    wandb_kwargs = {
        "project": config.wandb.project,
        "entity": config.wandb.entity,
        "config": flatten_nested_dict(config.to_dict()),
        "name": config.wandb.name if config.wandb.name else None,
        "mode": "online" if config.wandb.log else "disabled",
        "settings": wandb.Settings(code_dir=config.wandb.code_dir),
    }

    with wandb.init(**wandb_kwargs) as run:
        setup_training(run)
        # Load in the correct LR from sweeps
        new_vals = {}
        update_config_dict(config, run, new_vals)

        print(config)

        def forward_fn():
            flow = NICE(config.im_size**2, h_dim=config.hidden_dim)

            def _logpx(x):
                return flow.logpx(x)

            def _recons(x):
                return flow.reverse(flow.forward(x))

            def _sample():
                return flow.sample(config.batch_size)

            return _logpx, (_logpx, _recons, _sample)

        forward = hk.multi_transform(forward_fn)
        rng_seq = hk.PRNGSequence(config.seed)

        # load data
        ds = load_dataset(
            "train", config.batch_size, config.im_size, config.alpha, config.n_bits
        )
        ds_test = load_dataset(
            "test", config.batch_size, config.im_size, config.alpha, config.n_bits
        )

        # get init data
        x, _ = next(iter(ds))
        x = x.reshape(x.shape[0], -1)

        params = forward.init(next(rng_seq), x)
        logpx_fn, recons_fn, sample_fn = forward.apply

        print("Param shapes:")
        print(jax.tree_map(lambda x: x.shape, params))

        x_re = recons_fn(params, next(rng_seq), x)
        print(f"Recons Error: {((x - x_re)**2).mean()}")
        wandb.log({"recons_error": (np.array((x - x_re) ** 2).mean())})

        opt = optax.adam(config.lr)
        opt_state = opt.init(params)

        iteration = 0
        data_mean = x.mean(0)
        data_std = x.std(0)

        @functools.partial(jax.jit, static_argnums=3)
        def loss_fn(params, rng, x, with_wd=True):
            obj = logpx_fn(params, rng, x)

            l2_loss = 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(params))
            if with_wd:
                return -obj.mean() + config.weight_decay * l2_loss
            else:
                return -obj

        @jax.jit
        def update(params, opt_state, rng, x):
            loss, grad = jax.value_and_grad(loss_fn)(params, rng, x)

            updates, opt_state = opt.update(grad, opt_state)
            new_params = optax.apply_updates(params, updates)
            return loss, new_params, opt_state

        for epoch in range(config.num_epochs):
            for x, _ in iter(ds):
                x = x.reshape(x.shape[0], -1)
                loss, params, opt_state = update(params, opt_state, next(rng_seq), x)

                # print(params)

                if iteration % config.log_interval == 0:
                    print(f"Itr {iteration}, Epoch {epoch}, Loss {loss}")
                    wandb.log({"loss/train": (np.array(loss))})

                    x_re = recons_fn(params, jax.random.PRNGKey(0), x)
                    x_sample = sample_fn(params, jax.random.PRNGKey(12))
                    print(f"Recons Error: {((x - x_re)**2).mean()}")
                    wandb.log({"recons_error": (np.array((x - x_re) ** 2).mean())})

                    # make_grid(x, config.im_size, wandb_prefix="images/real")
                    # make_grid(x_re, config.im_size, wandb_prefix="images/recons")
                    # make_grid(x_sample, config.im_size, wandb_prefix="images/sample")

                iteration += 1

            test_loss = 0.0
            n_seen = 0
            for x, _ in iter(ds_test):
                x = x.reshape(x.shape[0], -1)
                loss = loss_fn(params, next(rng_seq), x, with_wd=False)
                test_loss += loss.sum()
                n_seen += loss.shape[0]
            test_loss = test_loss / n_seen
            wandb.log({"loss/test": (np.array(test_loss))})
            print(f"Epoch {epoch}, Test loss {test_loss}")

        cwd = os.getcwd()
        config.unlock()
        config.savedir = os.path.join(
            cwd, "saved_models", f"{config.alpha}_{config.n_bits}_{config.im_size}"
        )
        config.lock()

        if not os.path.exists(config.savedir):
            os.makedirs(config.savedir)

        if config.wandb.log_artifact:
            artifact_name = f"{config.alpha}_{config.n_bits}_{config.im_size}"

            artifact = wandb.Artifact(
                artifact_name,
                type="nice_params",
                metadata={
                    **{
                        "alpha": config.alpha,
                        "n_bits": config.n_bits,
                        "im_size": config.im_size,
                    }
                },
            )

            # Save model
            with artifact.new_file("params.pkl", "wb") as f:
                pickle.dump(params, f)

            wandb.log_artifact(artifact)


if __name__ == "__main__":
    os.environ["WANDB_API_KEY"] = "9835d6db89010f73306f92bb9a080c9751b25d28"

    jax.config.config_with_absl()

    def _main(argv):
        config = get_config()
        main(config)

    app.run(_main)