import logging
from pathlib import Path
import getpass


def config_logging(logger=None,
                   log_level=None,
                   log_format=f'%(asctime)s {getpass.getuser()}:%(name)s:%(levelname)s: %(message)s',
                   log_console=True,
                   log_path=None):
    print("Configuring logging")

    # Set logger
    if not isinstance(logger, logging.Logger):
        logger = logging.getLogger(logger)

    if log_level is None:
        log_level = logging.WARNING
    if isinstance(log_level, str):
        log_level = logging.getLevelName(log_level.upper())

    if log_format is None:
        log_format = '%(asctime)s %(name)s:%(levelname)s: %(message)s'

    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

    # Generic settings
    logger.setLevel(log_level)
    formatter = logging.Formatter(log_format)

    # Remove existing handlers
    # while logger.hasHandlers():
    while len(logger.handlers) > 0:
        print("Removing handler: {}".format(str(logger.handlers[0])))
        logger.removeHandler(logger.handlers[0])

    # Create console handler
    if log_console:
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        logger.info("Added console handler")

    if log_path is not None:
        # Create file handler
        fh = logging.FileHandler(str(log_path), encoding="UTF-8")
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.info("Added file handler: {}".format(log_path))


def log_dict(logger, d, name, level=logging.DEBUG):
    """Recursively log the keys and values of a dict."""
    for k, v in d.items():
        if isinstance(v, dict):
            log_dict(logger, v, name=f'{name}.{k}', level=level)
        else:
            logger.log(level, f'{name}.{k}: {v}')
