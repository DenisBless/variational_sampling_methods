import os

import ml_collections

LR_DICT = {
    "log_sonar": {
        "MCD_CAIS_UHA_sn": 1e-3,
        "MCD_CAIS_sn": 1e-3,
        "MCD_CAIS_var_sn": 1e-3,
        "MCD_U_a-lp-sn": 1e-3,
        "UHA": 1e-4,
        "MCD_ULA_sn": 1e-3,
        "MCD_ULA": 1e-4,
    },
    "log_ionosphere": {
        "MCD_CAIS_UHA_sn": 1e-3,
        "MCD_CAIS_sn": 1e-4,
        "MCD_U_a-lp-sn": 1e-3,
        "UHA": 1e-4,
        "MCD_ULA_sn": 1e-3,
        "MCD_ULA": 1e-4,
    },
    "lorenz": {
        "MCD_CAIS_UHA_sn": 1e-3,
        "MCD_CAIS_sn": 1e-5,
        "MCD_U_a-lp-sn": 1e-3,
        "UHA": 1e-3,
        "MCD_ULA_sn": 1e-5,
        "MCD_ULA": 1e-5,
    },
    "brownian": {
        "MCD_CAIS_UHA_sn": 1e-3,
        "MCD_CAIS_sn": 1e-3,
        "MCD_U_a-lp-sn": 1e-3,
        "UHA": 1e-4,
        "MCD_ULA_sn": 1e-4,
        "MCD_ULA": 1e-5,
    },
    "seeds": {
        "MCD_CAIS_UHA_sn": 1e-3,
        "MCD_CAIS_sn": 1e-3,
        "MCD_U_a-lp-sn": 1e-3,
        "UHA": 1e-3,
        "MCD_ULA_sn": 1e-3,
        "MCD_ULA": 1e-4,
    },
    "banana": {
        "MCD_CAIS_UHA_sn": 1e-3,
        "MCD_CAIS_sn": 1e-3,
        "MCD_U_a-lp-sn": 1e-3,
        "UHA": 1e-3,
        "MCD_ULA_sn": 1e-3,
        "MCD_ULA": 1e-4,
    },
    "lgcp": {
        "MCD_CAIS_UHA_sn": 1e-3,
        "MCD_CAIS_sn": 1e-4,
        "MCD_U_a-lp-sn": 1e-3,
        "UHA": 1e-4,
        "MCD_ULA_sn": 1e-4,
        "MCD_ULA": 1e-4,
    },
}

FUNNEL_EPS_DICT = {
    8: {"init_eps": 0.1, "lr": 0.01},
    16: {"init_eps": 0.1, "lr": 0.01},
    32: {"init_eps": 0.1, "lr": 0.005},
    64: {"init_eps": 0.1, "lr": 0.001},
    128: {"init_eps": 0.01, "lr": 0.01},
    256: {"init_eps": 0.01, "lr": 0.005},
}

TRACTABLE_DISTS = ["nice", "funnel", "gmm", "many_gmm"]


def get_config():
    config = ml_collections.ConfigDict()
    config.boundmode = "UHA"
    config.model = "lorenz"
    config.N = 5  # 5 for all except NICE
    config.nbridges = 8
    config.lfsteps = 1

    config.emb_dim = 20

    # "geffner" arch actually just hardcodes nlayers=2.
    config.nlayers = 3

    config.init_eta = 0.0
    config.init_eps = 1e-5
    config.init_sigma = 1.0
    config.pretrain_mfvi = True

    config.train_vi = True
    config.train_eps = True
    config.train_betas = True

    config.nn_arch = "geffner"  # "dds", "dds_grad"
    # This only affects dds and dds_grad nn_archs
    # In order to control size of "geffner" arch, use emb_dim.
    config.fully_connected_units = [64, 64]

    config.eps_schedule = ""
    config.grad_clipping = False

    config.mfvi_iters = 150000
    config.mfvi_lr = 0.01
    config.iters = 150000  # 150000 for all except NICE
    config.lr = 0.0001
    config.seed = 1
    config.id = -1
    config.run_cluster = 0
    config.n_samples = 500
    config.n_sinkhorn = 300
    config.n_input_dist_seeds = 30
    config.emb_dim = 20
    config.nlayers = 3

    config.use_ema = False

    # NICE Config/
    config.im_size = 14
    config.alpha = 0.05
    config.n_bits = 3
    config.hidden_dim = 1000

    # Funnel configs
    config.funnel_d = 10
    config.funnel_sig = 3
    config.funnel_clipy = 11

    # LGCP configs
    config.use_whitened = False

    # Many GMM configs
    config.gmm_easy_mode = False
    if config.gmm_easy_mode:
        config.n_mixes = 4
        config.loc_scaling = 10
    else:
        config.n_mixes = 40
        config.loc_scaling = 40

    cwd = os.getcwd()
    config.file_path = os.path.join(cwd, "../pines.csv")

    # Wandb Configs
    config.wandb = ml_collections.ConfigDict()
    config.wandb.log = True
    config.wandb.project = "final_cmcd"
    config.wandb.entity = "shreyaspadhy"
    config.wandb.code_dir = cwd
    config.wandb.name = ""
    config.wandb.log_artifact = True

    return config
