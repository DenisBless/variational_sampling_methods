"""
Annealed Flow Transport (AFT)
For further details see https://arxiv.org/abs/2102.07501
Code builds on https://github.com/google-deepmind/annealed_flow_transport
"""
import jax.numpy as jnp

from algorithms.aft import aft
from algorithms.common import flow_transport, markov_kernel, flows
from algorithms.common.utils import get_optimizer
import distrax
import haiku as hk
import jax


def aft_trainer(cfg, target):
    final_log_density = target.log_prob
    dim = target.dim
    alg_cfg = cfg.algorithm
    mcmc_cfg = cfg.algorithm.mcmc
    flow_cfg = cfg.algorithm.flows

    key = jax.random.PRNGKey(cfg.seed)

    initial_density = distrax.MultivariateNormalDiag(jnp.ones(dim) * alg_cfg.init_mean,
                                                     jnp.ones(dim) * alg_cfg.init_std)

    log_density_initial = initial_density.log_prob
    initial_sampler = initial_density.sample

    num_temps = alg_cfg.num_temps
    density_by_step = flow_transport.GeometricAnnealingSchedule(log_density_initial, final_log_density, num_temps)
    markov_kernel_by_step = markov_kernel.MarkovTransitionKernel(mcmc_cfg, density_by_step, num_temps)

    def flow_func(x):
        flow = getattr(flows, flow_cfg.flow_type)(flow_cfg)
        return flow(x)

    def inv_flow_func(x):
        flow = getattr(flows, flow_cfg.flow_type)(flow_cfg)
        return flow.inverse(x)

    flow_cfg.num_elem = dim
    flow_cfg.sample_shape = (dim, )

    flow_forward_fn = hk.without_apply_rng(hk.transform(flow_func))
    flow_inverse_fn = hk.without_apply_rng(hk.transform(inv_flow_func))

    key, subkey = jax.random.split(key)

    samples = initial_sampler(seed=subkey, sample_shape=(alg_cfg.batch_size,))
    key, subkey = jax.random.split(key)
    flow_init_params = flow_forward_fn.init(subkey,
                                            samples)

    opt = get_optimizer(alg_cfg.step_size, None)
    opt_init_state = opt.init(flow_init_params)
    # Add a log_step_output function here to enable non-trivial step logging.
    results = aft.outer_loop_aft(opt_update=opt.update,
                                 opt_init_state=opt_init_state,
                                 flow_init_params=flow_init_params,
                                 flow_apply=flow_forward_fn.apply,
                                 flow_inv_apply=flow_inverse_fn.apply,
                                 density_by_step=density_by_step,
                                 markov_kernel_by_step=markov_kernel_by_step,
                                 initial_sampler=initial_sampler,
                                 key=key,
                                 target=target,
                                 cfg=cfg)

    return results
