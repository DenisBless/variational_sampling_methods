"""Taken from https://github.com/google-deepmind/annealed_flow_transport/blob/master/annealed_flow_transport/densities.py."""

from typing import List

import os.path as osp
import pathlib
import itertools
import jax
import jax.numpy as jnp
import jax.scipy.linalg as slinalg
import numpy as np
import chex
import matplotlib.pyplot as plt

from targets.base_target import Target
from utils.path_utils import project_path

Array = chex.Array
"""Code for cox process density utilities."""

# TypeDefs
NpArray = np.ndarray
Array = chex.Array


def get_bin_counts(array_in: NpArray,
                   num_bins_per_dim: int) -> NpArray:
    """Divide two dimensional input space into a grid and count points in each.

  Point on the upper edge, which does happen in the data, go into the lower bin.
  The occurrence of these points is an artefact of the rescaling done on data.

  Args:
    array_in: (num_points,2) containing points in square [0,1]^2
    num_bins_per_dim: the number of bins per dimension for the grid.

  Returns:
    Numpy array of shape containing (num_bins_per_dim, num_bins_per_dim) counts.
  """
    chex.assert_rank(array_in, 2)
    scaled_array = array_in * num_bins_per_dim
    counts = np.zeros((num_bins_per_dim, num_bins_per_dim))
    for elem in scaled_array:
        flt_row, col_row = np.floor(elem)
        row = int(flt_row)
        col = int(col_row)
        # Deal with the case where the point lies exactly on upper/rightmost edge.
        if row == num_bins_per_dim:
            row -= 1
        if col == num_bins_per_dim:
            col -= 1
        counts[row, col] += 1
    return counts


def get_bin_vals(num_bins: int) -> NpArray:
    grid_indices = jnp.arange(num_bins)
    bin_vals = jnp.array([
        jnp.array(elem) for elem in itertools.product(grid_indices, grid_indices)
    ])
    return bin_vals


def gram(kernel, xs: Array) -> Array:
    """Given a kernel function and an array of points compute a gram matrix."""
    return jax.vmap(lambda x: jax.vmap(lambda y: kernel(x, y))(xs))(xs)


def kernel_func(x: Array,
                y: Array,
                signal_variance: Array,
                num_grid_per_dim: int,
                raw_length_scale: Array) -> Array:
    """Compute covariance/kernel function.

  K(m,n) = signal_variance * exp(-|m-n|/(num_grid_per_dim*raw_length_scale))

  Args:
    x: First point shape (num_spatial_dim,)
    y: Second point shape (num_spatial_dim,)
    signal_variance: non-negative scalar.
    num_grid_per_dim: Number of grid points per spatial dimension.
    raw_length_scale: Length scale of the undiscretized process.

  Returns:
    Scalar value of covariance function.
  """
    chex.assert_equal_shape([x, y])
    chex.assert_rank(x, 1)
    normalized_distance = jnp.linalg.norm(x - y, 2) / (
            num_grid_per_dim * raw_length_scale)
    return signal_variance * jnp.exp(-normalized_distance)


def poisson_process_log_likelihood(latent_function: Array,
                                   bin_area: Array,
                                   flat_bin_counts: Array) -> Array:
    """Discretized Poisson process log likelihood.

  Args:
    latent_function: Intensity per unit area of shape (total_dimensions,)
    bin_area: Scalar bin_area.
    flat_bin_counts: Non negative integer counts of shape (total_dimensions,)

  Returns:
    Total log likelihood of points.
  """
    chex.assert_rank([latent_function, bin_area], [1, 0])
    chex.assert_equal_shape([latent_function, flat_bin_counts])
    first_term = latent_function * flat_bin_counts
    second_term = -bin_area * jnp.exp(latent_function)
    return jnp.sum(first_term + second_term)


def get_latents_from_white(white: Array, const_mean: Array,
                           cholesky_gram: Array) -> Array:
    """Get latents from whitened representation.

  Let f = L e + mu where e is distributed as standard multivariate normal.
  Then Cov[f] = LL^T .
  In the present case L is assumed to be lower triangular and is given by
  the input cholesky_gram.
  mu_zero is a constant so that mu_i = const_mean for all i.

  Args:
    white: shape (total_dimensions,) e.g. (900,) for a 30x30 grid.
    const_mean: scalar.
    cholesky_gram: shape (total_dimensions, total_dimensions)

  Returns:
    points in the whitened space of shape (total_dimensions,)
  """
    chex.assert_rank([white, const_mean, cholesky_gram], [1, 0, 2])
    latent_function = jnp.matmul(cholesky_gram, white) + const_mean
    chex.assert_equal_shape([latent_function, white])
    return latent_function


def get_white_from_latents(latents: Array,
                           const_mean: Array,
                           cholesky_gram: Array) -> Array:
    """Get whitened representation from function representation.

  Let f = L e + mu where e is distributed as standard multivariate normal.
  Then Cov[f] = LL^T and e = L^-1(f-mu).
  In the present case L is assumed to be lower triangular and is given by
  the input cholesky_gram.
  mu_zero is a constant so that mu_i = const_mean for all i.

  Args:
    latents: shape (total_dimensions,) e.g. (900,) for a 30x30 grid.
    const_mean: scalar.
    cholesky_gram: shape (total_dimensions, total_dimensions)

  Returns:
    points in the whitened space of shape (total_dimensions,)
  """
    chex.assert_rank([latents, const_mean, cholesky_gram], [1, 0, 2])
    white = slinalg.solve_triangular(
        cholesky_gram, latents - const_mean, lower=True)
    chex.assert_equal_shape([latents, white])
    return white


class LogGaussianCoxPines(Target):
    """Log Gaussian Cox process posterior in 2D for pine saplings data.

  This follows Heng et al 2020 https://arxiv.org/abs/1708.08396 .

  config.file_path should point to a csv file of num_points columns
  and 2 rows containg the Finnish pines data.

  config.use_whitened is a boolean specifying whether or not to use a
  reparameterization in terms of the Cholesky decomposition of the prior.
  See Section G.4 of https://arxiv.org/abs/2102.07501 for more detail.
  The experiments in the paper have this set to False.

  num_dim should be the square of the lattice sites per dimension.
  So for a 40 x 40 grid num_dim should be 1600.
  """

    def __init__(self, num_grid_per_dim=40, use_whitened: bool = False, log_Z=None, can_sample=False, sample_bounds=None) -> None:
        assert num_grid_per_dim in (32, 40)
        super().__init__(dim=num_grid_per_dim**2, log_Z=log_Z, can_sample=can_sample)

        # Discretization is as in Controlled Sequential Monte Carlo
        # by Heng et al 2017 https://arxiv.org/abs/1708.08396
        self._num_latents = num_grid_per_dim ** 2
        self._num_grid_per_dim = num_grid_per_dim

        file_path = project_path('targets/data/pines.csv')
        pines_array = self.get_pines_points(file_path)[1:, 1:]
        bin_counts = jnp.array(
            get_bin_counts(pines_array,
                           self._num_grid_per_dim))

        self._flat_bin_counts = jnp.reshape(bin_counts, (self._num_latents))

        # This normalizes by the number of elements in the grid
        self._poisson_a = 1. / self._num_latents
        # Parameters for LGCP are as estimated in Moller et al, 1998
        # "Log Gaussian Cox processes" and are also used in Heng et al.

        self._signal_variance = 1.91
        self._beta = 1. / 33

        self._bin_vals = get_bin_vals(self._num_grid_per_dim)

        def short_kernel_func(x, y):
            return kernel_func(x, y, self._signal_variance,
                               self._num_grid_per_dim, self._beta)

        self._gram_matrix = gram(short_kernel_func, self._bin_vals)
        self._cholesky_gram = jnp.linalg.cholesky(self._gram_matrix)
        self._white_gaussian_log_normalizer = -0.5 * self._num_latents * jnp.log(
            2. * jnp.pi)

        half_log_det_gram = jnp.sum(jnp.log(jnp.abs(jnp.diag(self._cholesky_gram))))
        self._unwhitened_gaussian_log_normalizer = -0.5 * self._num_latents * jnp.log(
            2. * jnp.pi) - half_log_det_gram
        # The mean function is a constant with value mu_zero.
        self._mu_zero = jnp.log(126.) - 0.5 * self._signal_variance

        if use_whitened:
            self._posterior_log_density = self.whitened_posterior_log_density
        else:
            self._posterior_log_density = self.unwhitened_posterior_log_density

    def get_pines_points(self, file_path):
        """Get the pines data points."""
        with open(file_path, "rt") as input_file:
            b = np.genfromtxt(input_file, delimiter=",")
        return b

    def whitened_posterior_log_density(self, white: Array) -> Array:
        quadratic_term = -0.5 * jnp.sum(white ** 2)
        prior_log_density = self._white_gaussian_log_normalizer + quadratic_term
        latent_function = get_latents_from_white(white, self._mu_zero,
                                                 self._cholesky_gram)
        log_likelihood = poisson_process_log_likelihood(
            latent_function, self._poisson_a, self._flat_bin_counts)
        return prior_log_density + log_likelihood

    def unwhitened_posterior_log_density(self, latents: Array) -> Array:
        white = get_white_from_latents(latents, self._mu_zero,
                                       self._cholesky_gram)
        prior_log_density = -0.5 * jnp.sum(
            white * white) + self._unwhitened_gaussian_log_normalizer
        log_likelihood = poisson_process_log_likelihood(
            latents, self._poisson_a, self._flat_bin_counts)
        return prior_log_density + log_likelihood

    def log_prob(self, x: Array) -> Array:
        if x.ndim == 1:
            return self._posterior_log_density(x)
        else:
            assert x.ndim == 2
            return jax.vmap(self._posterior_log_density)(x)

    def visualise(self, samples: chex.Array = None, axes=None, show=False, prefix='') -> dict:
        return {}

    def sample(self, seed: chex.PRNGKey, sample_shape: chex.Shape) -> chex.Array:
        return None


CoxDist = LogGaussianCoxPines
