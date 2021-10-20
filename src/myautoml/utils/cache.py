import logging
import os
from pathlib import Path
from pickle import dump, load
import shutil
import uuid

from .pickle import load_pickle, save_pickle

_logger = logging.getLogger(__name__)


class HashableDict(dict):
    def __hash__(self):
        return hash(tuple(sorted(self.items())))


# We create a class CachePaths, so that it can set the cache dir automatically
# using an environment variable, which can be loaded dynamically from a .env file
# (i.e. after module imports have completed)
# The class is instantiated at the bottom of this file.
class CachePaths:
    def __init__(self):
        self._myautoml_cache_dir = None
        self._myautoml_cache_index_path = None
        self._myautoml_cache_data_dir = None

    @property
    def cache_dir(self):
        if self._myautoml_cache_dir is None:
            self.set_cache_dir()
        return self._myautoml_cache_dir

    @property
    def index_path(self):
        return self.cache_dir / 'index'

    @property
    def data_dir(self):
        return self.cache_dir / 'data'

    def set_cache_dir(self):
        cache_dir = os.getenv("MYAUTOML_CACHE_DIR", default=None)
        if cache_dir is None:
            # TODO: Should this be relative to the current working dir or home dir?
            # cache_dir = Path.home() / '.cache' / 'myautoml'
            cache_dir = Path('cache') / 'myautoml'
            _logger.warning(f"No cache dir specified, using default: {cache_dir}")
        else:
            cache_dir = Path(cache_dir)
            _logger.debug(f"Setting cache dir: {cache_dir}")
        self._myautoml_cache_dir = cache_dir


def get_cache_index():
    if cache_paths.index_path.exists():
        with open(cache_paths.index_path, 'rb') as index_file:
            cache_index = load(index_file)
    else:
        cache_index = {}
    return cache_index


def _save_cache_index(cache_index):
    cache_paths.index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_paths.index_path, 'wb') as index_file:
        dump(cache_index, index_file)


def delete_cache():
    _logger.warning(f"Deleting cache: {cache_paths.cache_dir}")
    shutil.rmtree(cache_paths.cache_dir)


def cached(func, cache_key=None, reset=False):
    """
    This wrapper loads the result of the function (given its args and kwargs) from a cache file if it exists.
    If not, it executes the function as usual and stores the result in a cache file for future reference.
    If the reset flag is set to True, any pre-existing cache file will be ignored and overwritten.

    Example 1:
        If you normally would call
            data = my_function(*args, **kwargs)
        Then you would now call
            data = cached(my_function)(*args, **kwargs)

    :param func:        The original function to be executed
    :param cache_key    The key to be used in the index. If None, a key will be created based on the function name, and
                        its args and kwargs
    :param reset:       Boolean indicator to specify whether the existing cache should be overwritten
    :return:            A modified function, which loads the result from the cache file if it exists,
                        and which executes the original function if it doesn't.
    """
    cache_index = get_cache_index()

    def cached_func(*args, **kwargs):
        if cache_key is None:
            ck = (('func', func.__name__),
                  ('args', args),
                  ('kwargs', *[(k, v) for k, v in kwargs.items()]))
        else:
            ck = cache_key
        _logger.debug(f"Cache key: {ck}")
        if ck not in cache_index.keys():
            cache_index[ck] = str(cache_paths.data_dir / uuid.uuid4().hex)
        cache_path = Path(cache_index[ck])

        if reset:
            _logger.info("Deleting any pre-existing cache")
            cache_path.unlink()

        if Path(cache_path).exists():
            return load_pickle(cache_path)
        result = func(*args, **kwargs)

        save_pickle(result, cache_path)
        _save_cache_index(cache_index)

        return result

    return cached_func


cache_paths = CachePaths()
