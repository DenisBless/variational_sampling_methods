"""NICE TARGET
"""
import functools
import os
import math
import pickle
from typing import Optional, List

import chex
import distrax
import jax
import jax.numpy as jnp
import numpy as np
import wandb
from flax import linen as nn
import matplotlib.pyplot as plt
from flax.training import checkpoints

from targets.base_target import Target
from utils.mode_classifier.classifier_model import CNN, FashionMnistCNN
from utils.path_utils import project_path

Array = jax.Array


class NICE(nn.Module):
    dim: int
    n_steps: int = 4
    h_depth: int = 5
    h_dim: int = 1000
    name: Optional[str] = None

    def setup(self):
        nets = []
        for _ in range(self.n_steps):
            layers = []
            for j in range(self.h_depth):
                if j != self.h_depth - 1:
                    layers.append(nn.Dense(self.h_dim))
                    layers.append(nn.relu)
                else:
                    layers.append(nn.Dense(self.dim // 2))
            net = nn.Sequential(layers)
            nets.append(net)

        self.logscale = self.param("logscale", nn.initializers.zeros, (self.dim,))
        self.nets = nets

    def __call__(self, x):
        chex.assert_shape(x, (None, self.dim))
        split = self.dim // 2
        if self.dim % 2 == 1:
            split += 1
        part = jnp.array(list(reversed(range(self.dim))))
        for net in self.nets:
            x_shuff = x[:, part]
            xa, xb = x_shuff[:, :split], x_shuff[:, split:]
            ya = xa
            yb = xb + net(xa)
            x = jnp.concatenate([ya, yb], -1)

        chex.assert_shape(x, (None, self.dim))
        return x

    def reverse(self, y):
        """Runs the model y->x."""
        chex.assert_shape(y, (None, self.dim))

        split = self.dim // 2
        if self.dim % 2 == 1:
            split += 1

        inv_part = jnp.array(list(reversed(range(self.dim))))
        for net in reversed(self.nets):
            ya, yb = y[:, :split], y[:, split:]
            xa = ya
            xb = yb - net(xa)
            x_shuff = jnp.concatenate([xa, xb], -1)
            y = x_shuff[:, inv_part]

        chex.assert_shape(y, (None, self.dim))
        return y

    def logpx(self, x):
        """Returns logp(x)."""
        z = self(x)
        zs = z * jnp.exp(self.logscale)[None, :]

        pz = distrax.MultivariateNormalDiag(jnp.zeros_like(zs), jnp.ones_like(zs))
        logp = pz.log_prob(zs)
        logp = logp + jnp.sum(self.logscale)

        chex.assert_shape(logp, (x.shape[0],))
        return logp

    def sample(self, key, n) -> Array:
        """Draws n samples from model."""
        key, subkey = jax.random.split(key)

        zs = jax.random.normal(subkey, (n, self.dim))
        z = zs / jnp.exp(self.logscale)[None, :]
        x = self.reverse(z)

        chex.assert_shape(x, (n, self.dim))
        return x

    def reparameterized_sample(self, zs: Array) -> Array:
        """Draws n samples from model."""
        z = zs / jnp.exp(self.logscale)[None, :]
        x = self.reverse(z)

        chex.assert_shape(x, zs.shape)
        return x

    def loss(self, x: Array) -> Array:
        """Loss function for training."""
        return -self.logpx(x)


def load_model_nice(dataset: str, dim: int):
    # all nice model are trained with alpha = 0.05, n_bits=3, and hidden_dim=1000

    im_size = int(jnp.sqrt(dim))

    pickle_file = project_path() + "/targets/data/" + f"params_nice_{dataset}_{im_size}x{im_size}_flax.pkl"
    loaded_params = pickle.load(open(pickle_file, "rb"))

    model = NICE(dim=dim)
    logpx_fn = lambda x: model.apply(loaded_params, x, method=model.logpx)
    sample_fn = lambda key, batch_size: model.apply(loaded_params, key, batch_size, method=model.sample)

    return logpx_fn, None, sample_fn


def classify(x, cnn, params, im_size):
    logits = cnn.apply({'params': params}, x.reshape((-1, im_size, im_size, 1)))
    return jnp.argmax(logits, -1)


class NiceTarget(Target):
    def __init__(self, dim, dataset="mnist", log_Z=0., can_sample=True, sample_bounds=None) -> None:
        super().__init__(dim, log_Z, can_sample)
        self.data_ndim = dim
        self.im_size = int(np.sqrt(dim))

        self.logpx_fn_without_rng, _, self.sample_fn_clean = load_model_nice(dataset, dim)

        state = checkpoints.restore_checkpoint(ckpt_dir=project_path('utils/mode_classifier'), target=None,
                                               prefix="{}_{}x{}_classifier_checkpoint".format(dataset,
                                                                                              int(math.sqrt(dim)),
                                                                                              int(math.sqrt(dim))))

        if dataset == "mnist":
            classifier = CNN()
        elif dataset == "fashion_mnist":
            classifier = FashionMnistCNN()
        else:
            raise NotImplementedError

        self.classify = jax.jit(
            functools.partial(classify, cnn=classifier, params=state['params'], im_size=self.im_size))

    def get_dim(self):
        return self.dim

    def log_prob(self, x: chex.Array) -> chex.Array:
        batched = x.ndim == 2

        if not batched:
            x = x[None,]

        log_prob = self.logpx_fn_without_rng(x)

        if not batched:
            log_prob = jnp.squeeze(log_prob, axis=0)

        return log_prob

    def sample(self, seed: chex.PRNGKey, sample_shape: chex.Shape) -> chex.Array:
        if len(sample_shape) == 0:
            n_samples = 1
            return self.sample_fn_clean(seed, n_samples)[0]
        else:
            n_samples = sample_shape[0]
            return self.sample_fn_clean(seed, n_samples)

    def entropy(self, samples: chex.Array = None):
        idx = self.classify(samples)
        unique_elements, counts = jnp.unique(idx, return_counts=True)
        mode_dist = counts / samples.shape[0]
        entropy = -jnp.sum(mode_dist * (jnp.log(mode_dist) / jnp.log(10)))
        return entropy

    def visualise(self,
                  samples: chex.Array,
                  axes: List[plt.Axes] = None,
                  show=False, prefix='') -> None:
        plt.close()
        n = 64
        x = np.array(samples[:n].reshape(-1, self.im_size, self.im_size))

        n_rows = int(np.sqrt(n))
        fig, ax = plt.subplots(n_rows, n_rows, figsize=(8, 8))

        # Plot each image
        for i in range(n_rows):
            for j in range(n_rows):
                ax[i, j].imshow(x[i * n_rows + j], cmap='gray')
                ax[i, j].axis('off')

        # plt.savefig(os.path.join(project_path('./figures/'), f"{prefix}nice.pdf"), bbox_inches='tight', pad_inches=0.1)
        # Log into wandb
        wb = {"figures/vis": [wandb.Image(fig)]}
        if show:
            plt.show()

        return wb


if __name__ == '__main__':
    # nice = NiceTarget(dim=196)
    # key = jax.random.PRNGKey(2)
    # samples = nice.sample(key, (64,))
    # # print(nice.entropy(samples))
    # nice.visualise(samples, show=True)
    # zeros = jnp.zeros([1000, 784])
    # lp =jax.jit(jax.vmap(nice.log_prob, in_axes=(0,)))
    # # print(lp(zeros))
    # algs = ['gmmvi_jax', 'smc', 'aft', 'craft', 'fab', 'mcd', 'ldvi', 'pis2', 'dis', 'dds2', 'gsb']
    algs = ['mfvi', 'gmmvi_jax', 'smc', 'aft', 'craft', 'fab', 'mcd', 'ldvi', 'pis2', 'dis', 'dds2', 'gsb']
    algs = ['gmmvi_jax', 'smc', 'aft', 'craft', 'fab', 'mcd', 'ldvi', 'pis2', 'dis', 'dds2', 'gsb']
    algs = ['gmmvi_jax']
    #### Generate Digit Visualizations
    nice = NiceTarget(dim=196)
    for alg in algs:
        path = project_path(f'samples/digits/samples_{alg}_nice_digits_196D_seed1.npy')
        samples = jnp.load(path)
        nice.visualise(samples, show=True, prefix=f'{alg}_digits')

    # #### Generate Fashion Visualizations
    # nice = NiceTarget(dataset='fashion_mnist', dim=784)
    # for alg in algs:
    #     path = project_path(f'samples/fashion/samples_{alg}_nice_fashion_784D_seed0.npy')
    #     samples = jnp.load(path)
    #     nice.visualise(samples, show=True, prefix=f'{alg}_fashion')
