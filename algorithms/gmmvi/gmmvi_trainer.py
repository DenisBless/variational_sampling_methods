from time import time

import wandb

from algorithms.common.eval_methods.tractable_density_methods import get_eval_fn
from algorithms.common.eval_methods.utils import extract_last_entry
from algorithms.gmmvi.optimization.gmmvi import setup_gmmvi
import jax

from utils.print_util import print_results

"""
Code for Gaussian Mixture Model Variational Inference (GMMVI).
Code migrated to Jax from https://github.com/OlegArenz/gmmvi.
"""


def gmmvi_trainer(cfg, target):
    key = jax.random.PRNGKey(cfg["seed"])
    target_samples = target.sample(jax.random.PRNGKey(0), (cfg.eval_samples,))
    eval_fn, logger = get_eval_fn(cfg, target, target_samples)

    gmmvi = setup_gmmvi(cfg, target, key)

    state = gmmvi.initial_train_state
    timer = 0
    logger = {}
    eval_freq = max(cfg.algorithm.iters // cfg.n_evals, 1)

    for step in range(0, cfg.algorithm.iters):
        iter_time = time()
        key, subkey = jax.random.split(key)
        state = gmmvi.train_iter(state, subkey)
        timer += time() - iter_time
        if (step % eval_freq == 0) or (step == cfg.algorithm.iters - 1):
            key, subkey = jax.random.split(key)
            logger = eval_fn(*gmmvi.eval(subkey, state, target_samples))
            logger["stats/step"].append(step)
            logger["stats/wallclock"].append(timer)
            logger['stats/num_samples'] = [state.sample_db_state.num_samples_written]
            logger['stats/num_components'] = [state.model_state.gmm_state.num_components]

            print_results(step, logger, cfg)

            if cfg.use_wandb:
                wandb.log(extract_last_entry(logger))

            print(f"{step}/{cfg.algorithm.iters}: "
                  f"The model now has {state.model_state.gmm_state.num_components} components ")

    return logger
