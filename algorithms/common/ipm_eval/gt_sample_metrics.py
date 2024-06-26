import os

import hydra
import jax.random
import numpy as np
from omegaconf import DictConfig

from algorithms.common.ipm_eval import discrepancies

"""
Small script for computing the distance between two ground truth sets of samples from the target.
"""

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


@hydra.main(version_base=None, config_path="../../../configs", config_name="base_conf")
def main(cfg: DictConfig) -> None:
    os.environ['HYDRA_FULL_ERROR'] = '1'
    cfg = hydra.utils.instantiate(cfg)

    target = cfg.target.fn
    d = 'sd'
    n_samples = 2000

    n_seeds = 20

    discrepancy = np.zeros(n_seeds)

    key, subkey = jax.random.split(jax.random.PRNGKey(1))
    for seed in range(n_seeds):
        groundtruth1 = target.sample(seed=jax.random.PRNGKey(0), sample_shape=(n_samples,))
        key, subkey = jax.random.split(key)
        groundtruth2 = target.sample(seed=subkey, sample_shape=(n_samples,))

        discrepancy[seed] = getattr(discrepancies, f'compute_{d}')(gt_samples=groundtruth1, samples=groundtruth2,
                                                                   config=cfg.target)
        print(discrepancy[seed])

    print('-------------------')
    print(
        bcolors.WARNING + f"& ${round(discrepancy.mean(0), 3)} \scriptstyle \pm {round(discrepancy.std(0), 3)}$" + bcolors.ENDC)


if __name__ == "__main__":
    main()
