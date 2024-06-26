"""Code builds on https://github.com/lollcat/fab-jax"""
from typing import Callable, Optional, NamedTuple, Any, Union
from functools import partial
import chex
import jax
import jax.numpy as jnp
from algorithms.fab.train.evaluate import setup_fab_eval_function
from algorithms.fab.train import build_fab_no_buffer_init_step_fns, \
    TrainStateNoBuffer, build_fab_with_buffer_init_step_fns, TrainStateWithBuffer
from algorithms.fab.buffer.prioritised_buffer import build_prioritised_buffer, PrioritisedBuffer
from algorithms.fab.flow import build_flow, FlowDistConfig
from algorithms.fab.sampling import build_smc, build_blackjax_hmc, simple_resampling, \
    build_metropolis, default_point_is_valid_fn, point_is_valid_if_in_bounds_fn
from algorithms.fab.utils.optimize import get_optimizer, OptimizerConfig
from targets.base_target import Target
import optax


class TrainingState(NamedTuple):
    params: Any
    opt_state: optax.OptState
    key: chex.PRNGKey


class TrainConfig(NamedTuple):
    n_iteration: int
    batch_size: int
    seed: int
    eval_freq: int
    init_state: Callable
    update: Callable
    eval_and_plot_fn: Callable
    save_model: bool
    use_wandb: bool
    verbose: bool


def setup_plotter(flow, smc, target: Target, plot_batch_size, buffer: Optional[PrioritisedBuffer] = None):
    @jax.jit
    def get_data_for_plotting(state: Union[TrainStateNoBuffer, TrainStateWithBuffer], key: chex.PRNGKey):
        x0 = flow.sample_apply(state.flow_params, key, (plot_batch_size,))

        def log_q_fn(x: chex.Array) -> chex.Array:
            return flow.log_prob_apply(state.flow_params, x)

        point, log_w, smc_state, smc_info = smc.step(x0, state.smc_state, log_q_fn, target.log_prob)
        x_smc = point.x
        _, x_smc_resampled = simple_resampling(key, log_w, x_smc)

        if buffer is not None:
            x_buffer = buffer.sample(key, state.buffer_state, plot_batch_size)[0]
        else:
            x_buffer = None

        return x0, x_smc, x_smc_resampled, x_buffer

    def plot(state: Union[TrainStateNoBuffer, TrainStateWithBuffer], key: chex.PRNGKey):
        x0, x_smc, x_smc_resampled, x_buffer = get_data_for_plotting(state, key)
        target.visualise(x_smc)

    return plot


def setup_fab_config(cfg, target) -> TrainConfig:
    dim = target.dim
    alg_cfg = cfg.algorithm

    # Setup buffer.
    buffer_max_length = alg_cfg.training.batch_size * alg_cfg.buffer.buffer_max_length_in_batches
    buffer_min_length = alg_cfg.training.batch_size * alg_cfg.buffer.buffer_min_length_in_batches
    n_updates_per_smc_forward_pass = alg_cfg.buffer.n_updates_per_smc_forward_pass
    w_adjust_clip = jnp.inf if alg_cfg.w_adjust_clip is None else alg_cfg.w_adjust_clip

    flow = build_flow(FlowDistConfig(dim=dim, **alg_cfg.flow))

    opt_cfg = dict(alg_cfg.training.optimizer)
    n_iter_warmup = opt_cfg.pop('warmup_n_epoch') * alg_cfg.buffer.n_updates_per_smc_forward_pass
    n_iter_total = alg_cfg.training.n_epoch * alg_cfg.buffer.n_updates_per_smc_forward_pass
    optimizer_config = OptimizerConfig(**opt_cfg,
                                       n_iter_total=n_iter_total,
                                       n_iter_warmup=n_iter_warmup)

    # Setup smc.
    if alg_cfg.smc.transition_operator == 'hmc':
        transition_operator = build_blackjax_hmc(
            dim=dim,
            n_outer_steps=alg_cfg.smc.hmc.n_outer_steps,
            init_step_size=alg_cfg.smc.hmc.init_step_size,
            target_p_accept=alg_cfg.smc.hmc.target_p_accept,
            adapt_step_size=alg_cfg.smc.hmc.tune_step_size,
            n_inner_steps=alg_cfg.smc.hmc.n_inner_steps)
    elif alg_cfg.smc.transition_operator == "metropolis":
        transition_operator = build_metropolis(dim, alg_cfg.smc.metropolis.n_outer_steps,
                                               alg_cfg.smc.metropolis.init_step_size,
                                               target_p_accept=alg_cfg.smc.metropolis.target_p_accept,
                                               tune_step_size=alg_cfg.smc.metropolis.tune_step_size)
    else:
        raise NotImplementedError

    if alg_cfg.smc.point_is_valid_fn_type == "in_bounds":
        point_is_valid_fn = partial(point_is_valid_if_in_bounds_fn,
                                    min_bounds=alg_cfg.smc.point_is_valid_fn.in_bounds.min,
                                    max_bounds=alg_cfg.smc.point_is_valid_fn.in_bounds.max
                                    )
    else:
        point_is_valid_fn = default_point_is_valid_fn

    smc = build_smc(transition_operator=transition_operator,
                    n_intermediate_distributions=alg_cfg.smc.n_intermediate_distributions,
                    spacing_type=alg_cfg.smc.spacing_type, alpha=alg_cfg.alpha,
                    use_resampling=alg_cfg.smc.use_resampling, point_is_valid_fn=point_is_valid_fn)

    # Optimizer
    optimizer, lr = get_optimizer(optimizer_config)

    # Prioritized buffer
    if alg_cfg.buffer.with_buffer:
        buffer = build_prioritised_buffer(dim=dim, max_length=buffer_max_length, min_length_to_sample=buffer_min_length)
    else:
        buffer = None
        n_updates_per_smc_forward_pass = None

    # Plotter
    # plot = setup_plotter(flow=flow, smc=smc, target=target, plot_batch_size=cfg.eval_samples, buffer=buffer)

    # Eval function
    # Eval uses AIS, and sets alpha=1 which is equivalent to targetting p.
    ais_eval = build_smc(
        transition_operator=transition_operator,
        n_intermediate_distributions=alg_cfg.smc.n_intermediate_distributions,
        spacing_type=alg_cfg.smc.spacing_type, alpha=1., use_resampling=False,
        point_is_valid_fn=point_is_valid_fn
    )

    eval_fn = setup_fab_eval_function(flow, ais_eval, target, cfg)

    def eval_and_plot_fn(state, subkey):
        info = eval_fn(state, subkey)
        return info

    if alg_cfg.buffer.with_buffer:
        assert buffer is not None and n_updates_per_smc_forward_pass is not None
        init, step = build_fab_with_buffer_init_step_fns(
            flow=flow, log_p_fn=target.log_prob,
            smc=smc, optimizer=optimizer,
            batch_size=alg_cfg.training.batch_size,
            buffer=buffer, n_updates_per_smc_forward_pass=n_updates_per_smc_forward_pass,
            w_adjust_clip=w_adjust_clip,
            use_reverse_kl_loss=alg_cfg.use_kl_loss
        )
    else:
        init, step = build_fab_no_buffer_init_step_fns(
            flow, log_p_fn=target.log_prob,
            smc=smc, optimizer=optimizer,
            batch_size=alg_cfg.training.batch_size)

    train_config = TrainConfig(n_iteration=alg_cfg.training.n_epoch,
                               batch_size=alg_cfg.training.batch_size,
                               seed=cfg.seed,
                               eval_freq=max(alg_cfg.training.n_epoch // cfg.n_evals, 1),
                               init_state=init,
                               update=step,
                               eval_and_plot_fn=eval_and_plot_fn,
                               save_model=cfg.save_model,
                               use_wandb=cfg.use_wandb,
                               verbose=cfg.verbose)
    return train_config
