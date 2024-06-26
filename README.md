# A Benchmark for Variational Sampling Methods

This repository accompanies the paper "[Beyond ELBOs: A Large-Scale Evaluation of Variational Methods for Sampling](https://arxiv.org/abs/2406.07423). [[`ICML'24`](https://openreview.net/forum?id=fVg9YrSllr),[`BibTeX`](#references)]" It includes a variety of sampling algorithms, such as sequential importance sampling algorithms and diffusion-based methods. A complete list of the available algorithms is provided below. Additionally, we offer several target densities, both low- and high-dimensional, with varying levels of complexity.
## Disclaimer
The codebase is currently under development. We are in the process of unifying various parts of the code, 
including migrating algorithms from the [Haiku](https://github.com/google-deepmind/dm-haiku) library to [Flax](https://github.com/google/flax). As a result, 
some outcomes may differ from those reported in the paper. 
These will be updated once the majority of the codebase is complete.

## Available Algorithms
The table below provides a overview of all available algorithms.

| **Acronym**               | **Method**                        | **Reference**                                                                                                                          |
|---------------------------|-----------------------------------|----------------------------------------------------------------------------------------------------------------------------------------|
| [MFVI](algorithms/mfvi)   | Gaussian Mean-Field VI            | [Bishop, 2006](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf) | 
| [GMMVI](algorithms/gmmvi) | Gaussian Mixture Model VI         | [Arenz et al., 2022](https://arxiv.org/abs/2209.11533)                                                                                 | 
| [NFVI](algorithms/nfvi)   | Normalizing Flow VI               | [Rezende & Mohamed, 2015](https://arxiv.org/abs/1505.05770)                                                                            |
| [SMC](algorithms/smc)     | Sequential Monte Carlo            | [Del Moral et al., 2006](https://www.stats.ox.ac.uk/~doucet/delmoral_doucet_jasra_sequentialmontecarlosamplersJRSSB.pdf)               | 
| [AFT](algorithms/aft)     | Annealed Flow Transport           | [Arbel et al., 2021](https://arxiv.org/abs/2102.07501)                                                                                 | 
| [CRAFT](algorithms/craft) | Continual Repeated AFT            | [Matthews et al., 2022](https://arxiv.org/abs/2201.13117)                                                                              |
| [FAB](algorithms/fab)     | Flow Annealed IS Bootstrap        | [Midgley et al., 2022](https://arxiv.org/abs/2208.01893)                                                                               | 
| [ULA](algorithms/ula)     | Uncorrected Langevin Annealing    | [Thin et al., 2021](https://arxiv.org/abs/2106.15921)                                                                                  | 
| [MCD](algorithms/mcd)     | Monte Carlo Diffusion             | [Doucet et al., 2022](https://arxiv.org/abs/2208.07698)                                                                                | 
| [UHA](algorithms/uha)     | Uncorrected Hamiltonian Annealing | [Geffner et al., 2021](https://arxiv.org/abs/2107.04150)                                                                               |
| [LDVI](algorithms/ldvi)   | Langevin Diffusion VI             | [Geffner et al., 2022](https://arxiv.org/abs/2208.07743)                                                                               | 
| [CMCD](algorithms/cmcd)   | Controlled MCD                    | [Vargas et al., 2023](https://arxiv.org/abs/2307.01050)                                                                                | 
| [PIS](algorithms/pis)     | Path Integral Sampler             | [Zhang et al., 2021](https://arxiv.org/abs/2111.15141)                                                                                 | 
| [DIS](algorithms/dis)     | Time-Reversed Diffusion Sampler   | [Berner et al., 2022](https://openreview.net/pdf?id=oYIjw37pTP)                                                                        | 
| [DDS](algorithms/dds)     | Denoising Diffusion Sampler       | [Vargas et al., 2023](https://arxiv.org/abs/2302.13834)                                                                                | 
| [GFN](algorithms/gfn)     | Generative Flow Networks          | [Lahlou et al., 2023](https://arxiv.org/abs/2301.12594)                                                                                | 
| [GBS](algorithms/gbs)     | General Bridge Sampler            | [Richter et al., 2023](https://arxiv.org/abs/2307.01198)                                                                               | 
The respective configuration files can be found [here](configs/algorithm).

## Available Target Densities
The table below provides a overview of available target densities. The 'ID' column provides identifier for running experiments 
via comand line. Further details in the [Running Experiments](#running-experiments) section.

|                                         | dim     | True log Z | Target Samples | ID                 |
|-----------------------------------------|---------|------------|----------------|--------------------|
| [**Funnel**](targets/funnel.py)         | 10      | ✔️         | ✔️             | funnel             |
| [**Credit**](targets/german_credit.py)  | 25      | ❌          | ❌              | credit             |
| [**Seeds**](targets/seeds.py)           | 26      | ❌          | ❌              | seeds              |
| [**Cancer**](targets/breast_cancer.py)  | 31      | ❌          | ❌              | cancer             |
| [**Brownian**](targets/brownian.py)     | 32      | ❌          | ❌              | brownian           |
| [**Ionosphere**](targets/ionosphere.py) | 35      | ❌          | ❌              | ionosphere         |
| [**Sonar**](targets/sonar.py)           | 61      | ❌          | ❌              | sonar              |
| [**Digits**](targets/nice.py)           | 196/784 | ✔️         | ✔️             | nice_digits        |
| [**Fashion**](targets/nice.py)          | 784     | ✔️         | ✔️             | nice_fashion       |
| [**LGCP**](targets/lgcp.py)             | 1600    | ❌          | ❌              | lgcp               |
| [**MoG**](targets/gmm40.py)             | any     | ✔️         | ✔️             | gaussian_mixture40 |
| [**MoS**](targets/student_t_mixture.py) | any     | ✔️         | ✔️             | student_t_mixture  |


The respective configuration files can be found [here](configs/target).

## Installation

First, clone the repo:

  ```
  git clone git@github.com:DenisBless/variational_sampling_methods.git
  cd variational_sampling_methods
  ```

We recommend using [Conda](https://conda.io/docs/user-guide/install/download.html) to set up the codebase:
  ```
  conda create -n sampling_bench python==3.10.14 pip --yes
  conda activate sampling_bench
  ```
Install the required packages using 
  ```
  pip install -r requirements.txt
  ```
Finally, we use [`wandb`](https://wandb.ai/) for experiment tracking. Login to your wandb account:
  ```
  wandb login
  ```
  You can also omit this step and add the `use_wandb=False` command line arg to your runs.


## Running Experiments

### Configuration
We use [`hydra`](https://hydra.cc/) for config management. The [base configuration](configs/base_conf.yaml) file sets 
parameters that are agnostic to the specific choice of algorithm and target density. The `wandb` entity can be set in the [setup config file](configs/setup.yaml).
### Running a single Experiment
In the simplest case, a single run can be started using
  ```
  python run.py algorithm=<algorithm ID> target=<target ID>
  ```
The algorithm ID is identical to the Acronym in the [algorithm table](#available-algorithms). The target ID can be found in the ID column of the [target table](#available-target-densities).
### Running multiple Experiments
Running multiple experiments can be done by using the hydra multi-run flag `-m/--multirun` flag.
For instance, running multiple seeds can be done via
  ```
  python run.py -m seed=0,1,2,3  algorithm=<algorithm ID> target=<target ID>
  ```
Using comma separation can also be used for running multiple algorithms/targets. 
### Running Experiments on a Cluster via Slurm
Running experiments on a cluster using [Slurm](https://slurm.schedmd.com/documentation.html) can be done via  
  ```
  python run.py +launcher=slurm algorithm=<algorithm ID> target=<target ID>
  ```
which uses the [slurm config](configs/launcher/slurm.yaml) config. Please make sure that you adapt the default settings to your slurm configuration.


## References

If you use parts of this codebase in your research, please cite us using the following BibTeX entries.

```
@misc{blessing2024elbos,
      title={Beyond ELBOs: A Large-Scale Evaluation of Variational Methods for Sampling}, 
      author={Denis Blessing and Xiaogang Jia and Johannes Esslinger and Francisco Vargas and Gerhard Neumann},
      year={2024},
      eprint={2406.07423},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Acknowledgements
Portions of the project are adapted from other repositories (as mentioned in the code): 
- https://github.com/shreyaspadhy/CMCD is licensed under MIT,
- https://github.com/OlegArenz/gmmvi is licensed under MIT,
- https://github.com/lollcat/fab-jax is licensed under MIT,
- https://github.com/tomsons22/LDVI is licensed under MIT,
- https://github.com/juliusberner/sde_sampler is licensed under MIT,
- https://github.com/franciscovargas/denoising_diffusion_samplers is licensed under MIT,
- https://github.com/antoninschrab/mmdfuse is licensed under MIT,
- https://github.com/google-deepmind/annealed_flow_transport is licensed under Apache-2.0. 
