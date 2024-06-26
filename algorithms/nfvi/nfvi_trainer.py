"""
Code for Normalizing Flow Variational Inference (NFVI).
For further details see: https://arxiv.org/abs/1505.05770
Code builds on https://github.com/google-deepmind/annealed_flow_transport
"""
import jax.numpy as jnp

from algorithms.common import flows
from algorithms.common.utils import get_optimizer
import distrax
import haiku as hk
import jax

from algorithms.nfvi import nfvi


def nfvi_trainer(cfg, target):
    dim = target.dim
    alg_cfg = cfg.algorithm
    flow_cfg = cfg.algorithm.flows
    key = jax.random.PRNGKey(cfg.seed)

    initial_density = distrax.MultivariateNormalDiag(jnp.ones(dim) * alg_cfg.init_mean,
                                                     jnp.ones(dim) * alg_cfg.init_std)

    log_density_initial = initial_density.log_prob
    initial_sampler = initial_density.sample

    def flow_func(x):
        flow = getattr(flows, flow_cfg.flow_type)(flow_cfg)
        return flow(x)

    def inv_flow_func(x):
        flow = getattr(flows, flow_cfg.flow_type)(flow_cfg)
        return flow.inverse(x)

    flow_cfg.num_elem = dim
    flow_cfg.sample_shape = (dim,)

    flow_forward_fn = hk.without_apply_rng(hk.transform(flow_func))
    flow_inverse_fn = hk.without_apply_rng(hk.transform(inv_flow_func))
    key, subkey = jax.random.split(key)

    samples = initial_sampler(seed=subkey, sample_shape=(alg_cfg.batch_size,))
    key, subkey = jax.random.split(key)
    flow_init_params = flow_forward_fn.init(subkey,
                                            samples)

    opt = get_optimizer(alg_cfg.step_size, None)
    opt_init_state = opt.init(flow_init_params)
    nfvi.outer_loop_vi(initial_sampler=initial_sampler,
                       opt_update=opt.update,
                       opt_init_state=opt_init_state,
                       flow_init_params=flow_init_params,
                       flow_apply=flow_forward_fn.apply,
                       flow_inverse_apply=flow_inverse_fn.apply,
                       initial_log_density=log_density_initial,
                       target=target,
                       cfg=cfg,
                       save_checkpoint=None)
