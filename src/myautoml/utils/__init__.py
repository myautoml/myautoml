from collections.abc import Mapping
import logging.config
import os
from pathlib import Path

from box import Box
from dotenv import load_dotenv
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
    print(f"Loading configuration file: {config_path}")
    if Path('.env').is_file():
        _logger.warning('Existence of a .env file may produce unexpected side-effects when loading the configuration'
                        'with EnvYAML. Use for example a .env.general file instead.')
        print('WARNING: Existence of a .env file may produce unexpected side-effects when loading the configuration'
              '         with EnvYAML. Use for example a .env.general file instead.')

    config = Box(EnvYAML(config_path, include_environment=include_environment).export())

    if include_default_config:
        default_logging_config = Box(EnvYAML(Path(__file__).parent / 'default_config.yml',
                                             include_environment=False).export())
        config = recursive_update(default_logging_config, config)

    return config


def start_script(dotenv='.env.general',
                 config_yaml='config.yaml',
                 include_environment=True,
                 include_default_config=True):
    load_dotenv(dotenv)
    if Path(config_yaml).suffix not in [".yml", ".yaml"]:
        config_yaml = os.getenv(config_yaml)
    config = load_config(config_yaml,
                         include_environment=include_environment,
                         include_default_config=include_default_config)
    Path(config.logging.handlers.debug_file_handler.filename).parent.mkdir(parents=True, exist_ok=True)
    Path(config.logging.handlers.info_file_handler.filename).parent.mkdir(parents=True, exist_ok=True)
    Path(config.logging.handlers.warning_file_handler.filename).parent.mkdir(parents=True, exist_ok=True)
    Path(config.logging.handlers.error_file_handler.filename).parent.mkdir(parents=True, exist_ok=True)
    Path(config.logging.handlers.critical_file_handler.filename).parent.mkdir(parents=True, exist_ok=True)
    logging.config.dictConfig(config.logging)
    return config
