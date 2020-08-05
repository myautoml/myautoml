import logging

_logger = logging.getLogger(__name__)


def flatten_params(params):
    _logger.log(level=logging.NOTSET, msg="Flattening a dictionary of parameters")
    flattened = {}
    for k, v in params.items():
        if isinstance(v, dict):
            # Ignore the higher level and insert the flattened params below this key
            flattened.update(flatten_params(v))
        else:
            flattened[k] = v
    return flattened


def prep_params(params):
    _logger.log(level=logging.NOTSET, msg="Ensuring integer parameters to be integers")
    # Numeric values loaded from a configuration.yaml file tend to have dtype float.
    for p in params.keys():
        if isinstance(params[p], float):
            if params[p] == int(params[p]):
                params[p] = int(params[p])
    return params
