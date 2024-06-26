from algorithms.gmmvi.configs import get_default_algorithm_config, update_config
from algorithms.gmmvi.models.gmm import setup_diagonal_gmm, setup_full_cov_gmm
from algorithms.gmmvi.models.gmm_wrapper import GMMWrapperState, setup_gmm_wrapper
from algorithms.gmmvi.optimization.gmmvi_modules.component_stepsize_adaptation import (
    setup_improvement_based_stepsize_adaptation,
    setup_decaying_component_stepsize_adaptation, setup_fixed_component_stepsize_adaptation)
from algorithms.gmmvi.optimization.gmmvi_modules.component_adaptation import (
    ComponentAdaptationState, setup_fixed_component_adaptation, setup_vips_component_adaptation)
from algorithms.gmmvi.optimization.gmmvi_modules.ng_based_component_updater import (
    setup_kl_constrained_ng_based_component_updater,
    setup_direct_ng_based_component_updater, setup_ng_based_component_updater_iblr)
from algorithms.gmmvi.optimization.gmmvi_modules.ng_estimator import (
    NgEstimatorState, setup_stein_ng_estimator, setup_more_ng_estimator)
from algorithms.gmmvi.optimization.gmmvi_modules.sample_selector import (
    setup_vips_sample_selector, setup_lin_sample_selector, setup_fixed_sample_selector)
from algorithms.gmmvi.optimization.gmmvi_modules.weight_stepsize_adaptation import (
    WeightStepsizeAdaptationState, setup_fixed_weight_stepsize_adaptation, setup_decaying_weight_stepsize_adaptation,
    setup_improvement_based_weight_stepsize_adaptation)
from algorithms.gmmvi.optimization.gmmvi_modules.weight_updater import (
    setup_trust_region_based_weight_updater, setup_direct_weight_updater)
from algorithms.gmmvi.optimization.sample_db import SampleDBState, setup_sampledb
import jax
import jax.numpy as jnp
from typing import NamedTuple, Callable
import chex


class TrainState(NamedTuple):
    temperature: float
    num_updates: chex.Array
    model_state: GMMWrapperState
    sample_db_state: SampleDBState
    component_adaptation_state: ComponentAdaptationState
    ng_estimator_state: NgEstimatorState
    weight_stepsize_adapter_state: WeightStepsizeAdaptationState


class GMMVI(NamedTuple):
    initial_train_state: TrainState
    train_iter: Callable
    eval: Callable


def setup_gmmvi(config, target, seed):
    dim = target.dim
    target_log_prob = target.log_prob

    # get necessary config entries
    config = update_config(update_config(get_default_algorithm_config(config.algorithm.algorithm), config), config.algorithm)

    # setup GMM
    if config["model_initialization"]["use_diagonal_covs"]:
        gmm = setup_diagonal_gmm(dim)
    else:
        gmm = setup_full_cov_gmm(dim)
    gmm_state = gmm.init_gmm_state(seed,
                                   config["model_initialization"]["num_initial_components"],
                                   config["model_initialization"]["prior_mean"],
                                   config["model_initialization"]["prior_scale"],
                                   config["model_initialization"]["use_diagonal_covs"],
                                   config["model_initialization"]["init_std"] ** 2)

    if "initial_l2_regularizer" in config["ng_estimator_config"]:
        initial_l2_regularizer = config["ng_estimator_config"]['initial_l2_regularizer']
    else:
        initial_l2_regularizer = 1e-12

    # setup GMMWrapper
    model = setup_gmm_wrapper(gmm,
                              config["component_stepsize_adapter_config"]["initial_stepsize"],
                              initial_l2_regularizer,
                              10000)
    model_state = model.init_gmm_wrapper_state(gmm_state)

    # setup SampleDB
    sample_db = setup_sampledb(dim,
                               config["use_sample_database"],
                               config["max_database_size"],
                               config["model_initialization"]["use_diagonal_covs"],
                               config['sample_selector_config']["desired_samples_per_component"])
    sample_db_state = sample_db.init_sampleDB_state()

    # setup NgEstimator
    if config["ng_estimator_type"] == "Stein":
        ng_estimator = setup_stein_ng_estimator(model,
                                                dim,
                                                config["model_initialization"]["use_diagonal_covs"],
                                                config['ng_estimator_config']["only_use_own_samples"],
                                                config['ng_estimator_config']["use_self_normalized_importance_weights"],
                                                )
        ng_estimator_state = ng_estimator.init_ng_estimator_state()
    # elif config["ng_estimator_type"] == "MORE":
    #     quad_regression = setup_quad_regression(dim)
    #     quad_regression_state = quad_regression.init_quad_reg_state()
    #
    #     ng_estimator = setup_more_ng_estimator(model,
    #                                            quad_regression,
    #                                            dim,
    #                                            config['ng_estimator_config']["only_use_own_samples"],
    #                                            config['ng_estimator_config']["use_self_normalized_importance_weights"])
    #     ng_estimator_state = ng_estimator.init_ng_estimator_state(quad_regression_state)
    else:
        raise ValueError(f"config['ng_estimator_type'] is '{config['ng_estimator_type']}' "
                         f"which is an unknown type")

    # setup NgBasedComponentUpdater
    if config["ng_based_updater_type"] == "trust-region":
        ng_based_component_updater = setup_kl_constrained_ng_based_component_updater(model,
                                                                                     dim,
                                                                                     config["model_initialization"]["use_diagonal_covs"],
                                                                                     config['temperature'],
                                                                                     initial_l2_regularizer)
    # elif config["ng_based_updater_type"] == "direct":
    #     ng_based_component_updater = setup_direct_ng_based_component_updater(model)
    # elif config["ng_based_updater_type"] == "iBLR":
    #     ng_based_component_updater = setup_ng_based_component_updater_iblr(model)
    else:
        raise ValueError(
            f"config['ng_based_updater_type'] is '{config['ng_based_updater_type']}' which is an unknown type")

    # setup ComponentAdaptation
    if config["num_component_adapter_type"] == "adaptive":
        component_adapter = setup_vips_component_adaptation(sample_db,
                                                            model,
                                                            target_log_prob,
                                                            dim,
                                                            config["model_initialization"]["prior_mean"],
                                                            config["model_initialization"]["init_std"] ** 2,
                                                            config["model_initialization"]["use_diagonal_covs"],
                                                            config["num_component_adapter_config"]["del_iters"],
                                                            config["num_component_adapter_config"]["add_iters"],
                                                            config["num_component_adapter_config"]["max_components"],
                                                            config["num_component_adapter_config"]["thresholds_for_add_heuristic"],
                                                            config["num_component_adapter_config"]["min_weight_for_del_heuristic"],
                                                            config["num_component_adapter_config"]["num_database_samples"],
                                                            config["num_component_adapter_config"]["num_prior_samples"],
                                                            )
        component_adapter_state = component_adapter.init_component_adaptation()
    elif config["num_component_adapter_type"] == "fixed":
        component_adapter = setup_fixed_component_adaptation()
        component_adapter_state = component_adapter.init_component_adaptation()
    else:
        raise ValueError(
            f"config['num_component_adapter_type'] is '{config['num_component_adapter_type']}' "
            f"which is an unknown type")

    # setup Component StepsizeAdaptation
    if config["component_stepsize_adapter_type"] == "improvement-based":
        component_stepsize_adapter = setup_improvement_based_stepsize_adaptation(config["component_stepsize_adapter_config"]["min_stepsize"],
                                                                                 config["component_stepsize_adapter_config"]["max_stepsize"],
                                                                                 config["component_stepsize_adapter_config"]["stepsize_inc_factor"],
                                                                                 config["component_stepsize_adapter_config"]["stepsize_dec_factor"])
    elif config["component_stepsize_adapter_type"] == "decaying":
        component_stepsize_adapter = setup_decaying_component_stepsize_adaptation(config["component_stepsize_adapter_config"]["initial_stepsize"],
                                                                                  config["component_stepsize_adapter_config"]["annealing_exponent"])

    elif config["component_stepsize_adapter_type"] == "fixed":
        component_stepsize_adapter = setup_fixed_component_stepsize_adaptation()
    else:
        raise ValueError(
            f"config['component_stepsize_adapter_type'] is '{config['component_stepsize_adapter_type']}' "
            f"which is an unknown type")

    # setup SampleSelector
    if config["sample_selector_type"] == "fixed":
        sample_selector = setup_fixed_sample_selector(sample_db,
                                                      model,
                                                      target_log_prob,
                                                      config['sample_selector_config']["desired_samples_per_component"],
                                                      config['sample_selector_config']["ratio_reused_samples_to_desired"])
    elif config["sample_selector_type"] == "mixture-based":
        sample_selector = setup_lin_sample_selector(sample_db,
                                                    model,
                                                    target_log_prob,
                                                    config['sample_selector_config']["desired_samples_per_component"],
                                                    config['sample_selector_config']["ratio_reused_samples_to_desired"])

    elif config["sample_selector_type"] == "component-based":
        sample_selector = setup_vips_sample_selector(sample_db,
                                                     model,
                                                     target_log_prob,
                                                     config['sample_selector_config']["desired_samples_per_component"],
                                                     config['sample_selector_config']["ratio_reused_samples_to_desired"])
    else:
        raise ValueError(
            f"config['sample_selector_type'] is '{config['sample_selector_type']}' which is an unknown type")

    # setup weight_updater
    if config["weight_updater_type"] == "direct":
        weight_updater = setup_direct_weight_updater(model,
                                                     config['temperature'],
                                                     config["weight_updater_config"]["use_self_normalized_importance_weights"])
    elif config["weight_updater_type"] == "trust-region":
        weight_updater = setup_trust_region_based_weight_updater(model,
                                                                 config['temperature'],
                                                                 config["weight_updater_config"]["use_self_normalized_importance_weights"])
    else:
        raise ValueError(
            f"config['weight_updater_type'] is '{config['weight_updater_type']}' which is an unknown type")

    # setup WeightStepsizeAdaptation
    if config["weight_stepsize_adapter_type"] == "fixed":
        weight_stepsize_adapter = setup_fixed_weight_stepsize_adaptation()
        weight_stepsize_adapter_state = weight_stepsize_adapter.init_weight_stepsize_adaptation(config['weight_stepsize_adapter_config']["initial_stepsize"])
    elif config["weight_stepsize_adapter_type"] == "decaying":
        weight_stepsize_adapter = setup_decaying_weight_stepsize_adaptation(config['weight_stepsize_adapter_config']["initial_stepsize"],
                                                                            config['weight_stepsize_adapter_config']["annealing_exponent"])
        weight_stepsize_adapter_state = weight_stepsize_adapter.init_weight_stepsize_adaptation()
    elif config["weight_stepsize_adapter_type"] == "improvement_based":
        weight_stepsize_adapter = setup_improvement_based_weight_stepsize_adaptation(config['weight_stepsize_adapter_config']["min_stepsize"],
                                                                                     config['weight_stepsize_adapter_config']["max_stepsize"],
                                                                                     config['weight_stepsize_adapter_config']["stepsize_inc_factor"],
                                                                                     config['weight_stepsize_adapter_config']["stepsize_dec_factor"])
        weight_stepsize_adapter_state = weight_stepsize_adapter.init_weight_stepsize_adaptation(config['weight_stepsize_adapter_config']["initial_stepsize"])
    else:
        raise ValueError(
            f"config['weight_stepsize_adapter_type'] is '{config['weight_stepsize_adapter_type']}' "
            f"which is an unknown type")

    def train_iter(train_state: TrainState, key: chex.Array):

        key, subkey = jax.random.split(key)
        new_sample_db_state, samples, mapping, sample_dist_densities, target_lnpdfs, target_lnpdf_grads = sample_selector.select_samples(train_state.model_state,
                                                                                                                                         train_state.sample_db_state,
                                                                                                                                         subkey)
        new_component_stepsizes = component_stepsize_adapter.update_stepsize(train_state.model_state)
        new_model_state = model.update_stepsizes(train_state.model_state, new_component_stepsizes)
        expected_hessian_neg, expected_grad_neg = ng_estimator.get_expected_hessian_and_grad(new_model_state,
                                                                                             samples,
                                                                                             mapping,
                                                                                             sample_dist_densities,
                                                                                             target_lnpdfs,
                                                                                             target_lnpdf_grads,
                                                                                             int(train_state.model_state.gmm_state.num_components))
        new_model_state = ng_based_component_updater.apply_NG_update(new_model_state,
                                                                     expected_hessian_neg,
                                                                     expected_grad_neg,
                                                                     new_model_state.stepsizes)

        new_weight_stepsize_adapter_state = weight_stepsize_adapter.update_stepsize(train_state.weight_stepsize_adapter_state, new_model_state)
        new_model_state = weight_updater.update_weights(new_model_state, samples, sample_dist_densities, target_lnpdfs,
                                                        new_weight_stepsize_adapter_state.stepsize)
        new_num_updates = train_state.num_updates + 1
        key, subkey = jax.random.split(key)
        new_model_state, new_component_adapter_state, new_sample_db_state = component_adapter.adapt_number_of_components(train_state.component_adaptation_state,
                                                                                                    new_sample_db_state,
                                                                                                    new_model_state,
                                                                                                    new_num_updates,
                                                                                                    subkey)

        return TrainState(temperature=train_state.temperature,
                          model_state=new_model_state,
                          component_adaptation_state=new_component_adapter_state,
                          ng_estimator_state=train_state.ng_estimator_state,
                          num_updates=new_num_updates,
                          sample_db_state=new_sample_db_state,
                          weight_stepsize_adapter_state=new_weight_stepsize_adapter_state)

    def eval(seed: chex.Array, train_state: TrainState, target_samples=None):
        samples = model.sample(train_state.model_state.gmm_state, seed, config["eval_samples"])[0]
        log_prob_model = jax.vmap(model.log_density, in_axes=(None, 0))(train_state.model_state.gmm_state, samples)
        log_prob_target = jax.vmap(target.log_prob)(samples)
        log_ratio = log_prob_target - log_prob_model

        if target_samples is not None:
            fwd_log_prob_model = jax.vmap(model.log_density, in_axes=(None, 0))(train_state.model_state.gmm_state, target_samples)
            fwd_log_prob_target = jax.vmap(target.log_prob)(target_samples)
            fwd_log_ratio = fwd_log_prob_target - fwd_log_prob_model
        else:
            fwd_log_ratio = None

        return samples, log_ratio, log_prob_target, fwd_log_ratio

    initial_train_state = TrainState(temperature=config['temperature'],
                                     num_updates=jnp.array([0]),
                                     model_state=model_state,
                                     sample_db_state=sample_db_state,
                                     component_adaptation_state=component_adapter_state,
                                     ng_estimator_state=ng_estimator_state,
                                     weight_stepsize_adapter_state=weight_stepsize_adapter_state)

    return GMMVI(initial_train_state=initial_train_state, train_iter=train_iter, eval=eval)
