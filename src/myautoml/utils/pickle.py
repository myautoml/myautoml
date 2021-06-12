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


def cached(cache_path, func):
    """
    This wrapper adds an additional required positional argument cache_path to a function.

    If the cache_path points to an existing file, the contents are unpickled and returned.

    If the cache_path does not point to a file, then the original function is executed with the original parameters.
    The result is first saved as a pickle file in the cache_path, and then returned.

    Example 1:
        If you normally would call
            data = my_function(*args, **kwargs)
        Then you would now call
            data = cached('/path/to/my/cache', my_function)(*args, **kwargs)

    :param cache_path   The path to the cache file
    :param func:        The original function to be executed
    :return:            A modified function, which loads the result from the cache file if it exists,
                        and which executes the original function if it doesn't.
    """

    def cached_func(*args, **kwargs):
        if Path(cache_path).exists():
            return load_pickle(cache_path)
        result = func(*args, **kwargs)
        save_pickle(result, cache_path)
        return result

    return cached_func
