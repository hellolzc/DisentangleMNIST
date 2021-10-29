from typing import Dict
import json
import yaml


def parse_and_update(config, kwargs, prefix='', allow_add=False):
    """Update parameters through dict kwargs recursively
    Modify the config in place. Use copy.deepcopy if you want to keep it.
    """
    for k, v in kwargs.items():
        if isinstance(v, Dict):
            parse_and_update(config[k], v, '{}{}.'.format(prefix, k), allow_add=allow_add)
        else:
            if (k not in config) and (not allow_add):
                raise ValueError('ERROR: config has not attribut %s' % k)
            print("[INFO @ %s]"%__name__, 'Set {}{} to {}'.format(prefix, k, v))
            config[k] = v


def create_hparams(fpath='', yaml_hparams_string='', json_hparams_string='', debug_print=True, allow_add=False):
    """Load Hyper parameters and update parameters through dict kwargs recursively"""
    # Load
    if fpath !='':
        config = yaml.load(open(fpath, "r"), Loader=yaml.FullLoader)
    else:
        config = {}
    if debug_print:
        print("[INFO @ %s]"%__name__, "Load HParams from %s" % fpath)
    # Modify
    if (yaml_hparams_string is not None) and (yaml_hparams_string != ''):
        update_dict = yaml.load(yaml_hparams_string, Loader=yaml.FullLoader)
        if update_dict is not None:
            parse_and_update(config, update_dict, allow_add=allow_add)
    if (json_hparams_string is not None) and (json_hparams_string != ''):
        update_dict = json.loads(json_hparams_string)
        parse_and_update(config, update_dict, allow_add=allow_add)
    # Output
    if debug_print:
        print(yaml.dump(config))
    return config

def hparams_debug_string(config):
    return 'Hyperparameters:\n' + yaml.dump(config)


if __name__ == '__main__':
    import copy

    model_config = yaml.load(open('./config/LJSpeech/model.yaml', "r"), Loader=yaml.FullLoader)
    print('model_config:\n' + json.dumps(model_config, indent=2))

    # kwargs = json.loads("""
    # {
    #   "variance_embedding": {
    #     "pitch_quantization": "log",
    #     "energy_quantization": "log",
    #     "n_bins": 512
    #   }
    # }""")

    kwargs = yaml.load("""
variance_embedding:
    energy_quantization: log
    n_bins: 512
    pitch_quantization: log
""" , Loader=yaml.FullLoader)

    print('kwargs:\n' + json.dumps(kwargs, indent=2))

    new_config = copy.deepcopy(model_config)
    parse_and_update(new_config, kwargs)

    print('new_config:\n' + json.dumps(new_config, indent=2))
