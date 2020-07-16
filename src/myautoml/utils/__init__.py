from collections.abc import Mapping
import logging
from pathlib import Path

from box import Box
from envyaml import EnvYAML

_logger = logging.getLogger(__name__)


def recursive_update(d, u):
    """Updates a dictionary d with updates from u recursively."""
    for k, v in u.items():
        if isinstance(v, Mapping):
            d[k] = recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def load_config(config_path, include_environment=True, include_default_config=True):
    _logger.debug(f"Loading configuration file: {config_path}")
    if Path('.env').is_file():
        _logger.warning('Existence of a .env file may produce unexpected side-effects when loading the configuration'
                        'with EnvYAML. Use for example a .env.general file instead.')

    config = Box(EnvYAML(config_path, include_environment=include_environment).export())

    if include_default_config:
        default_logging_config = Box(EnvYAML(Path(__file__).parent / 'default_config.yml',
                                             include_environment=False).export())
        return recursive_update(default_logging_config, config)
    return config
