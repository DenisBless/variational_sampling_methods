import jax.numpy as jnp
from jax import vmap, jit
from functools import partial

"""
Code for computing the Maximum-Mean-Discrepancy (MMD) based on https://github.com/antoninschrab/mmdfuse.
"""


@partial(jit)
def mmd_median(
        X,
        Y,
):
    # Assertions
    m = X.shape[0]
    n = Y.shape[0]
    assert n >= 2 and m >= 2
    kernel = 'gaussian'
    l = "l2"
    # Compute kernel matrix
    Z = jnp.concatenate((X, Y))
    pairwise_matrix = jax_distances(Z, Z, l, matrix=True)
    distances = pairwise_matrix[jnp.triu_indices(pairwise_matrix.shape[0])]
    bandwidth = jnp.median(distances)

    d_XY = jax_distances(X, Y, l, matrix=True)
    d_XX = jax_distances(X, X, l, matrix=True)
    d_YY = jax_distances(Y, Y, l, matrix=True)
    K_XY = kernel_matrix(d_XY, l, kernel, bandwidth, 0.5)
    K_XX = kernel_matrix(d_XX, l, kernel, bandwidth, 0.5)
    K_YY = kernel_matrix(d_YY, l, kernel, bandwidth, 0.5)

    mmd = (K_XX.sum() / (n * (n - 1))) + (K_YY.sum() / (m * (m - 1))) - 2 * K_XY.mean()
    mmd = jnp.sqrt(jnp.maximum(1e-20, mmd))  # Ensure non-negative value
    return mmd


def kernel_matrix(pairwise_matrix, l, kernel, bandwidth, rq_kernel_exponent=0.5):
    """
    Compute kernel matrix for a given kernel and bandwidth.

    inputs: pairwise_matrix: (2m,2m) matrix of pairwise distances
            l: "l1" or "l2" or "l2sq"
            kernel: string from ("gaussian", "laplace", "imq", "matern_0.5_l1", "matern_1.5_l1", "matern_2.5_l1", "matern_3.5_l1", "matern_4.5_l1", "matern_0.5_l2", "matern_1.5_l2", "matern_2.5_l2", "matern_3.5_l2", "matern_4.5_l2")
    output: (2m,2m) pairwise distance matrix

    Warning: The pair of variables l and kernel must be valid.
    """
    d = pairwise_matrix / bandwidth
    if kernel == "gaussian" and l == "l2":
        return jnp.exp(-(d ** 2) / 2)
    elif kernel == "laplace" and l == "l1":
        return jnp.exp(-d * jnp.sqrt(2))
    elif kernel == "rq" and l == "l2":
        return (1 + d ** 2 / (2 * rq_kernel_exponent)) ** (-rq_kernel_exponent)
    elif kernel == "imq" and l == "l2":
        return (1 + d ** 2) ** (-0.5)
    elif (kernel == "matern_0.5_l1" and l == "l1") or (
            kernel == "matern_0.5_l2" and l == "l2"
    ):
        return jnp.exp(-d)
    elif (kernel == "matern_1.5_l1" and l == "l1") or (
            kernel == "matern_1.5_l2" and l == "l2"
    ):
        return (1 + jnp.sqrt(3) * d) * jnp.exp(-jnp.sqrt(3) * d)
    elif (kernel == "matern_2.5_l1" and l == "l1") or (
            kernel == "matern_2.5_l2" and l == "l2"
    ):
        return (1 + jnp.sqrt(5) * d + 5 / 3 * d ** 2) * jnp.exp(-jnp.sqrt(5) * d)
    elif (kernel == "matern_3.5_l1" and l == "l1") or (
            kernel == "matern_3.5_l2" and l == "l2"
    ):
        return (
                1 + jnp.sqrt(7) * d + 2 * 7 / 5 * d ** 2 + 7 * jnp.sqrt(7) / 3 / 5 * d ** 3
        ) * jnp.exp(-jnp.sqrt(7) * d)
    elif (kernel == "matern_4.5_l1" and l == "l1") or (
            kernel == "matern_4.5_l2" and l == "l2"
    ):
        return (
                1
                + 3 * d
                + 3 * (6 ** 2) / 28 * d ** 2
                + (6 ** 3) / 84 * d ** 3
                + (6 ** 4) / 1680 * d ** 4
        ) * jnp.exp(-3 * d)
    else:
        raise ValueError('The values of "l" and "kernel" are not valid.')


def jax_distances(X, Y, l, max_samples=None, matrix=False):
    if l == "l1":

        def dist(x, y):
            z = x - y
            return jnp.sum(jnp.abs(z))

    elif l == "l2":

        def dist(x, y):
            z = x - y
            return jnp.sqrt(jnp.sum(jnp.square(z)))

    else:
        raise ValueError("Value of 'l' must be either 'l1' or 'l2'.")
    vmapped_dist = vmap(dist, in_axes=(0, None))
    pairwise_dist = vmap(vmapped_dist, in_axes=(None, 0))
    output = pairwise_dist(X[:max_samples], Y[:max_samples])
    if matrix:
        return output
    else:
        return output[jnp.triu_indices(output.shape[0])]


@partial(jit, static_argnums=(2, 3, 4))
def compute_bandwidths(X, Y, l, number_bandwidths, only_median=False):
    Z = jnp.concatenate((X, Y))
    distances = jax_distances(Z, Z, l, matrix=False)
    median = jnp.median(distances)
    if only_median:
        return median
    distances = distances + (distances == 0) * median
    dd = jnp.sort(distances)
    lambda_min = dd[(jnp.floor(len(dd) * 0.05).astype(int))] / 2
    lambda_max = dd[(jnp.floor(len(dd) * 0.95).astype(int))] * 2
    bandwidths = jnp.linspace(lambda_min, lambda_max, number_bandwidths)
    return bandwidths
