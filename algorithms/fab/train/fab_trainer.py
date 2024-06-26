"""
Code for Flow Annealed Importance Sampling Bootstrap (FAB).
For further details see https://arxiv.org/abs/2208.01893.
Code builds on https://github.com/lollcat/fab-jax.
"""
import pickle
import jax
import wandb

from algorithms.common.eval_methods.sis_methods import get_eval_fn
from algorithms.common.eval_methods.utils import extract_last_entry
from algorithms.fab.train.setup_training import setup_fab_config
from time import time
from utils.path_utils import project_path, make_model_dir
from utils.print_util import print_results


def save_model(model_path, state, step):
    with open(project_path(f'{model_path}/{step}.pkl'), 'wb') as f:
        pickle.dump(state, f)


def load_model(model_path):
    with open(model_path, "rb") as f:
        state = pickle.load(f)
    return state


def fab_trainer(cfg, target):
    # setup fab
    config = setup_fab_config(cfg, target)
    alg_cfg = cfg.algorithm

    key = jax.random.PRNGKey(config.seed)
    key, subkey = jax.random.split(key)
    state = config.init_state(subkey)

    test_losses = []
    timer = 0

    ais_nfe = 2 * config.batch_size * (alg_cfg.smc.n_intermediate_distributions - 1)
    mcmc_nfe = config.batch_size * (alg_cfg.smc.n_intermediate_distributions - 1) * (
            alg_cfg.smc.hmc.n_outer_steps * alg_cfg.smc.hmc.n_inner_steps +
            alg_cfg.smc.metropolis.n_outer_steps)
    smc_nfe = ais_nfe + mcmc_nfe

    target_samples = target.sample(jax.random.PRNGKey(0), (cfg.eval_samples,))
    eval_fn, logger = get_eval_fn(cfg, target, target_samples)

    for iteration in range(config.n_iteration):
        iter_time = time()

        key, subkey = jax.random.split(key)

        # info contains further information that can be logged if necessary
        state, info = config.update(state)
        timer += time() - iter_time

        if (iteration % config.eval_freq == 0) or (iteration == config.n_iteration - 1):
            key, subkey = jax.random.split(key)
            logger = eval_fn(*config.eval_and_plot_fn(state, subkey))
            logger["stats/step"] = [iteration]
            logger["stats/wallclock"] = [timer]
            logger["stats/nfe"] = [(iteration + 1) * (config.batch_size + smc_nfe)]

            print_results(iteration, logger, config)

            if cfg.use_wandb:
                wandb.log(extract_last_entry(logger))

    return logger, test_losses
