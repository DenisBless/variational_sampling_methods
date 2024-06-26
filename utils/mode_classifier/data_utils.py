import tensorflow as tf
import tensorflow_datasets as tfds


def dequantize(x, y, n_bits=3):
    n_bins = 2.0 ** n_bits
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


def load_dataset(dataset: str, split: str, batch_size: int, im_size: int, alpha: float, n_bits: int):
    """Loads the dataset as a generator of batches."""
    ds, ds_info = tfds.load(name=dataset, split=split, as_supervised=True, with_info=True)
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
