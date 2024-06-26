import yaml
import os
from mergedeep import merge, Strategy

def load_yaml(filename):
    with open(filename, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
            return config
        except yaml.YAMLError as exc:
            print(exc)

def get_default_algorithm_config(algorithm_id):
    print(f"Using default parameters for codename {algorithm_id}")
    MODULE_CONF_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "module_configs")

    LETTER_TO_PATH = {
        "Z": os.path.join(MODULE_CONF_PATH, "ng_estimator", "MORE.yml"),
        "S": os.path.join(MODULE_CONF_PATH, "ng_estimator", "Stein.yml"),

        "A": os.path.join(MODULE_CONF_PATH, "component_adaptation", "adaptive.yml"),
        "E": os.path.join(MODULE_CONF_PATH, "component_adaptation", "fixed.yml"),

        "P": os.path.join(MODULE_CONF_PATH, "sample_selector", "mixture-based.yml"),
        "M": os.path.join(MODULE_CONF_PATH, "sample_selector", "component-based.yml"),
        "Q": os.path.join(MODULE_CONF_PATH, "sample_selector", "fixed.yml"),

        "I": os.path.join(MODULE_CONF_PATH, "ng_based_component_updater", "direct.yml"),
        "Y": os.path.join(MODULE_CONF_PATH, "ng_based_component_updater", "iBLR.yml"),
        "T": os.path.join(MODULE_CONF_PATH, "ng_based_component_updater", "trust-region.yml"),

        "F": os.path.join(MODULE_CONF_PATH, "component_stepsize_adaptation", "fixed.yml"),
        "D": os.path.join(MODULE_CONF_PATH, "component_stepsize_adaptation", "decaying.yml"),
        "R": os.path.join(MODULE_CONF_PATH, "component_stepsize_adaptation", "improvement-based.yml"),

        "U": os.path.join(MODULE_CONF_PATH, "weight_updater", "direct.yml"),
        "O": os.path.join(MODULE_CONF_PATH, "weight_updater", "trust-region.yml"),

        "X": os.path.join(MODULE_CONF_PATH, "weight_stepsize_adaptation", "fixed.yml"),
        "G": os.path.join(MODULE_CONF_PATH, "weight_stepsize_adaptation", "decaying.yml"),
        "N": os.path.join(MODULE_CONF_PATH, "weight_stepsize_adaptation", "improvement-based.yml"),
    }

    merged_config = dict()
    [merge(merged_config, load_yaml(LETTER_TO_PATH[letter.upper()])) for letter in algorithm_id]
    return merged_config


def update_config(default_values, updates):
    updated_dict = dict(default_values)
    return merge(updated_dict, updates, strategy=Strategy.REPLACE)
