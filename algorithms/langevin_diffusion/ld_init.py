from algorithms.langevin_diffusion import base_dist as bd
from jax._src.flatten_util import ravel_pytree

from algorithms.langevin_diffusion.score_network import initialize_score_network, initialize_pis_network
import jax
import jax.numpy as jnp


def initialize_ula(config, dim, base_dist_params=None):
    params_train = {}  # Has all trainable parameters
    params_notrain = {}  # Non trainable parameters

    trainable = config.trainable
    init_std = config.init_std
    num_temps = config.num_temps
    num_learned_betas = num_temps

    if "bd" in trainable:
        params_train["bd"] = base_dist_params
        if base_dist_params is None:
            params_train["bd"] = bd.initialize(dim, init_std)
    else:
        params_notrain["bd"] = base_dist_params
        if base_dist_params is None:
            params_notrain["bd"] = bd.initialize(dim, init_std)

    if "eps" in trainable:  # integrator step size
        params_train["eps"] = config.eps
    else:
        params_notrain["eps"] = config.eps

    # Everything related to the annealing schedule (betas)
    if num_temps < num_learned_betas:
        num_learned_betas = num_temps
    mgridref_y = jnp.ones(num_learned_betas + 1) * 1.0
    params_notrain["gridref_x"] = jnp.linspace(0, 1, num_learned_betas + 2)
    params_notrain["target_x"] = jnp.linspace(0, 1, num_temps + 2)[1:-1]
    if "betas" in trainable:
        params_train["mgridref_y"] = mgridref_y
    else:
        params_notrain["mgridref_y"] = mgridref_y

    # Other fixed parameters
    params_fixed = (dim, num_temps, None, 'ULA')
    params_flat, unflatten = ravel_pytree((params_train, params_notrain))
    return params_flat, unflatten, params_fixed


def initialize_mcd(config, dim, base_dist_params=None):
    params_train = {}  # Has all trainable parameters
    params_notrain = {}  # Non trainable parameters

    alg_cfg = config.algorithm
    trainable = alg_cfg.trainable
    init_std = alg_cfg.init_std
    num_temps = alg_cfg.num_temps
    num_learned_betas = num_temps

    if "bd" in trainable:
        params_train["bd"] = base_dist_params
        if base_dist_params is None:
            params_train["bd"] = bd.initialize(dim, init_std)
    else:
        params_notrain["bd"] = base_dist_params
        if base_dist_params is None:
            params_notrain["bd"] = bd.initialize(dim, init_std)

    if "eps" in trainable:  # integrator step size
        params_train["eps"] = alg_cfg.eps
    else:
        params_notrain["eps"] = alg_cfg.eps

    if alg_cfg.approx_network == "score":
        init_fun_approx_network, apply_fun_approx_network = initialize_score_network(
            dim, alg_cfg.score_network_emb_dim, num_temps, nlayers=alg_cfg.score_network_num_layer)
    else:
        init_fun_approx_network, apply_fun_approx_network = initialize_pis_network(dim,
                                                                                   alg_cfg.pis_network_fully_connected_units)

    params_train["approx_network"] = init_fun_approx_network(jax.random.PRNGKey(config.seed), None)[1]

    # Everything related to the annealing schedule (betas)
    if num_temps < num_learned_betas:
        num_learned_betas = num_temps
    mgridref_y = jnp.ones(num_learned_betas + 1) * 1.0
    params_notrain["gridref_x"] = jnp.linspace(0, 1, num_learned_betas + 2)
    params_notrain["target_x"] = jnp.linspace(0, 1, num_temps + 2)[1:-1]
    if "betas" in trainable:
        params_train["mgridref_y"] = mgridref_y
    else:
        params_notrain["mgridref_y"] = mgridref_y

    # Other fixed parameters
    params_fixed = (dim, num_temps, apply_fun_approx_network, 'MCD')
    params_flat, unflatten = ravel_pytree((params_train, params_notrain))
    return params_flat, unflatten, params_fixed


def initialize_uha(config, dim, base_dist_params=None):
    params_train = {}  # Has all trainable parameters
    params_notrain = {}  # Non trainable parameters

    trainable = config.trainable
    init_std = config.init_std
    num_temps = config.num_temps
    num_learned_betas = num_temps

    if "bd" in trainable:
        params_train["bd"] = base_dist_params
        if base_dist_params is None:
            params_train["bd"] = bd.initialize(dim, init_std)
    else:
        params_notrain["bd"] = base_dist_params
        if base_dist_params is None:
            params_notrain["bd"] = bd.initialize(dim, init_std)

    if "eps" in trainable:  # integrator step size
        params_train["eps"] = config.eps
    else:
        params_notrain["eps"] = config.eps

    if "gamma" in trainable:  # friction coefficient
        params_train["gamma"] = config.gamma
    else:
        params_notrain["gamma"] = config.gamma

    # Everything related to the annealing schedule (betas)
    if num_temps < num_learned_betas:
        num_learned_betas = num_temps
    mgridref_y = jnp.ones(num_learned_betas + 1) * 1.0
    params_notrain["gridref_x"] = jnp.linspace(0, 1, num_learned_betas + 2)
    params_notrain["target_x"] = jnp.linspace(0, 1, num_temps + 2)[1:-1]
    if "betas" in trainable:
        params_train["mgridref_y"] = mgridref_y
    else:
        params_notrain["mgridref_y"] = mgridref_y

    # Other fixed parameters
    params_fixed = (dim, num_temps, None, 'UHA')
    params_flat, unflatten = ravel_pytree((params_train, params_notrain))
    return params_flat, unflatten, params_fixed


def initialize_ldvi(config, dim, base_dist_params=None):
    params_train = {}  # Has all trainable parameters
    params_notrain = {}  # Non trainable parameters

    alg_cfg = config.algorithm
    trainable = alg_cfg.trainable
    init_std = alg_cfg.init_std
    num_temps = alg_cfg.num_temps
    num_learned_betas = num_temps

    if "bd" in trainable:
        params_train["bd"] = base_dist_params
        if base_dist_params is None:
            params_train["bd"] = bd.initialize(dim, init_std)
    else:
        params_notrain["bd"] = base_dist_params
        if base_dist_params is None:
            params_notrain["bd"] = bd.initialize(dim, init_std)

    if "eps" in trainable:  # integrator step size
        params_train["eps"] = alg_cfg.eps
    else:
        params_notrain["eps"] = alg_cfg.eps

    if "gamma" in trainable:  # friction coefficient
        params_train["gamma"] = alg_cfg.gamma
    else:
        params_notrain["gamma"] = alg_cfg.gamma

    if alg_cfg.approx_network == "score":
        init_fun_approx_network, apply_fun_approx_network = initialize_score_network(dim,
                                                                                     alg_cfg.score_network_emb_dim,
                                                                                     num_temps,
                                                                                     rho_dim=dim,
                                                                                     nlayers=alg_cfg.score_network_num_layer)
    else:
        init_fun_approx_network, apply_fun_approx_network = initialize_pis_network(
            dim, alg_cfg.pis_network_fully_connected_units, rho_dim=dim)

    params_train["approx_network"] = init_fun_approx_network(jax.random.PRNGKey(config.seed), None)[1]

    # Everything related to the annealing schedule (betas)
    if num_temps < num_learned_betas:
        num_learned_betas = num_temps
    mgridref_y = jnp.ones(num_learned_betas + 1) * 1.0
    params_notrain["gridref_x"] = jnp.linspace(0, 1, num_learned_betas + 2)
    params_notrain["target_x"] = jnp.linspace(0, 1, num_temps + 2)[1:-1]
    if "betas" in trainable:
        params_train["mgridref_y"] = mgridref_y
    else:
        params_notrain["mgridref_y"] = mgridref_y

    # Other fixed parameters
    params_fixed = (dim, num_temps, apply_fun_approx_network, 'LDVI')
    params_flat, unflatten = ravel_pytree((params_train, params_notrain))
    return params_flat, unflatten, params_fixed


def initialize_cmcd(config, dim, base_dist_params=None):
    params_train = {}  # Has all trainable parameters
    params_notrain = {}  # Non trainable parameters

    alg_cfg = config.algorithm
    trainable = alg_cfg.trainable
    init_std = alg_cfg.init_std
    num_temps = alg_cfg.num_temps
    num_learned_betas = num_temps

    if "bd" in trainable:
        params_train["bd"] = base_dist_params
        if base_dist_params is None:
            params_train["bd"] = bd.initialize(dim, init_std)
    else:
        params_notrain["bd"] = base_dist_params
        if base_dist_params is None:
            params_notrain["bd"] = bd.initialize(dim, init_std)

    if "eps" in trainable:  # integrator step size
        params_train["eps"] = alg_cfg.eps
    else:
        params_notrain["eps"] = alg_cfg.eps

    if alg_cfg.approx_network == "score":
        init_fun_approx_network, apply_fun_approx_network = initialize_score_network(
            dim, alg_cfg.score_network_emb_dim, num_temps,
            nlayers=alg_cfg.score_network_num_layer)
    else:
        init_fun_approx_network, apply_fun_approx_network = initialize_pis_network(dim,
                                                                                   alg_cfg.pis_network_fully_connected_units)

    params_train["approx_network"] = init_fun_approx_network(jax.random.PRNGKey(config.seed), None)[1]

    # Everything related to the annealing schedule (betas)
    if num_temps < num_learned_betas:
        num_learned_betas = num_temps
    mgridref_y = jnp.ones(num_learned_betas + 1) * 1.0
    params_notrain["gridref_x"] = jnp.linspace(0, 1, num_learned_betas + 2)
    params_notrain["target_x"] = jnp.linspace(0, 1, num_temps + 2)[1:-1]
    if "betas" in trainable:
        params_train["mgridref_y"] = mgridref_y
    else:
        params_notrain["mgridref_y"] = mgridref_y

    # Other fixed parameters
    params_fixed = (dim, num_temps, apply_fun_approx_network, 'CMCD')
    params_flat, unflatten = ravel_pytree((params_train, params_notrain))
    return params_flat, unflatten, params_fixed
