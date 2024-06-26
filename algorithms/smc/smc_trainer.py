import jax
import jax.numpy as jnp

from algorithms.common import flow_transport, markov_kernel
from algorithms.smc import smc
import distrax

"""
Code for Sequential Monte Carlo (SMC.
"""


def smc_trainer(cfg, target):
    final_log_density = target.log_prob
    dim = target.dim
    alg_cfg = cfg.algorithm
    mcmc_cfg = cfg.algorithm.mcmc

    initial_density = distrax.MultivariateNormalDiag(jnp.ones(dim) * alg_cfg.init_mean, jnp.ones(dim) * alg_cfg.init_std)

    log_density_initial = initial_density.log_prob
    initial_sampler = initial_density.sample

    num_temps = alg_cfg.num_temps
    density_by_step = flow_transport.GeometricAnnealingSchedule(log_density_initial, final_log_density, num_temps)
    markov_kernel_by_step = markov_kernel.MarkovTransitionKernel(mcmc_cfg, density_by_step, num_temps)

    key = jax.random.PRNGKey(cfg.seed)
    key, subkey = jax.random.split(key)

    smc.outer_loop_smc(density_by_step=density_by_step,
                       initial_sampler=initial_sampler,
                       markov_kernel_by_step=markov_kernel_by_step,
                       key=key,
                       target=target,
                       cfg=cfg)