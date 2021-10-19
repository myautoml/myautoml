import logging
from pathlib import Path
from pickle import dump, load

_logger = logging.getLogger(__name__)


def save_pickle(obj, path):
    filepath = Path(path)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    _logger.debug(f"Saving to pickle file: {filepath}")
    with open(filepath, 'wb') as cache_file:
        dump(obj, cache_file)
    _logger.debug(f"Object saved to pickle file: {filepath}")


def load_pickle(path):
    _logger.debug(f"Loading from pickle file: {path}")
    with open(path, 'rb') as cache_file:
        obj = load(cache_file)
    _logger.debug(f"Object loaded from pickle file: {path}")
    return obj
