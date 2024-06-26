import math

import jax
import jax.numpy as jnp  # JAX NumPy

from flax import linen as nn  # The Linen API
from flax.training import train_state, checkpoints
import optax  # The Optax gradient processing and optimization library

import numpy as np  # Ordinary NumPy
import tensorflow_datasets as tfds

from utils.mode_classifier.classifier_model import CNN, FashionMnistCNN
from utils.mode_classifier.data_utils import load_dataset
from utils.path_utils import project_path



def compute_metrics(logits, labels):
    loss = jnp.mean(optax.softmax_cross_entropy(logits, jax.nn.one_hot(labels, num_classes=10)))
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    metrics = {
        'loss': loss,
        'accuracy': accuracy
    }
    return metrics


def get_datasets(dataset):
    ds_builder = tfds.builder(dataset)
    ds_builder.download_and_prepare()
    # Split into training/test sets
    train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
    # Convert to floating-points
    train_ds['image'] = jnp.float32(train_ds['image']) / 255.0
    test_ds['image'] = jnp.float32(test_ds['image']) / 255.0
    return train_ds, test_ds


@jax.jit
def train_step(state, x, y):
    def loss_fn(params):
        logits = FashionMnistCNN().apply({'params': params}, x)
        loss = jnp.mean(optax.softmax_cross_entropy(
            logits=logits,
            labels=jax.nn.one_hot(y, num_classes=10)))
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = compute_metrics(logits, y)
    return state, metrics


@jax.jit
def eval_step(params, batch_x, batch_y):
    logits = FashionMnistCNN().apply({'params': params}, batch_x)
    return compute_metrics(logits, batch_y)


def train_epoch(state, train_ds, epoch):
    batch_metrics = []

    for x, y in iter(train_ds):
        state, metrics = train_step(state, x, y)
        batch_metrics.append(metrics)

    training_batch_metrics = jax.device_get(batch_metrics)
    training_epoch_metrics = {
        k: np.mean([metrics[k] for metrics in training_batch_metrics])
        for k in training_batch_metrics[0]}

    print('Training - epoch: %d, loss: %.4f, accuracy: %.2f' % (
        epoch, training_epoch_metrics['loss'], training_epoch_metrics['accuracy'] * 100))

    return state, training_epoch_metrics


def eval(state, test_ds, epoch):
    batch_metrics = []

    for x, y in iter(test_ds):
        batch_metrics.append(eval_step(state.params, x, y))

    test_batch_metrics = jax.device_get(batch_metrics)
    test_epoch_metrics = {
        k: np.mean([metrics[k] for metrics in test_batch_metrics])
        for k in test_batch_metrics[0]}

    print('Test - epoch: %d, loss: %.4f, accuracy: %.2f' % (
        epoch, test_epoch_metrics['loss'], test_epoch_metrics['accuracy'] * 100))

    return state, test_epoch_metrics


def eval_model(model, batch_x, batch_y):
    metrics = eval_step(model, batch_x, batch_y)
    metrics = jax.device_get(metrics)
    eval_summary = jax.tree_map(lambda x: x.item(), metrics)
    return eval_summary['loss'], eval_summary['accuracy']


if __name__ == '__main__':
    BATCHSIZE = 128
    IMSIZE = 28
    ALPHA = 0.05
    BITS = 3
    NUM_EPOCHS = 300
    SAVE_DIR = project_path('utils/mode_classifier/')

    DATASET = "fashion_mnist"

    ds = load_dataset(DATASET, "train", BATCHSIZE, IMSIZE, ALPHA, BITS)
    ds_test = load_dataset(DATASET, "train", BATCHSIZE, IMSIZE, ALPHA, BITS)
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)

    # set CNN in train_step and eval_step also!!
    if DATASET == "fashion_mnist":
        cnn = FashionMnistCNN()
    elif DATASET == "mnist":
        cnn= CNN()
    else:
        raise NotImplementedError
    nesterov_momentum = 0.9
    learning_rate = 0.001
    tx = optax.sgd(learning_rate=learning_rate, nesterov=nesterov_momentum)
    params = cnn.init(init_rng, jnp.ones([1, IMSIZE, IMSIZE, 1]))['params']
    state = train_state.TrainState.create(apply_fn=cnn.apply, params=params, tx=tx)

    num_epochs = NUM_EPOCHS
    batch_size = BATCHSIZE

    # nice_target = NiceTarget()
    # rng, init_rng = jax.random.split(rng)
    # target_samples = nice_target.sample(init_rng, (BATCHSIZE,))

    for epoch in range(1, num_epochs + 1):
        # Use a separate PRNG key to permute image data during shuffling
        rng, input_rng = jax.random.split(rng)
        # Run an optimization step over a training batch
        state, train_metrics = train_epoch(state, ds, epoch)
        # Evaluate on the test set after each training epoch
        eval(state, ds_test, epoch)

    checkpoints.save_checkpoint(ckpt_dir=SAVE_DIR, target=state, step=0, prefix="{}_{}x{}_classifier_checkpoint_".format(DATASET, IMSIZE, IMSIZE))

    # logits = CNN().apply({'params': state.params}, target_samples.reshape((-1, IMSIZE, IMSIZE, 1)))
    # idx = jnp.argmax(logits, -1)
    # import matplotlib.pyplot as plt
    #
    # for i in range(BATCHSIZE):
    #     plt.imshow(target_samples[i, :].reshape((IMSIZE, IMSIZE)))
    #     plt.title(str(idx[i]))
    #     plt.show()
