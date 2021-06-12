import logging
from pathlib import Path
import getpass


class LevelFormatter:
    """Logging Formatter to add colors and count warning / errors"""

    def __init__(self, fmt=None, datefmt=None, style='%', validate=True):
        if fmt is None:
            grey = "\x1b[38;21m"
            green = "\x1b[32;21m"
            yellow = "\x1b[33;21m"
            red = "\x1b[31;21m"
            bold_red = "\x1b[31;1m"
            reset = "\x1b[0m"
            fmt_str = "%(asctime)s - %(name)s (%(filename)s:%(lineno)d) - %(levelname)s - %(message)s"
            fmt = {
                logging.DEBUG: grey + fmt_str + reset,
                logging.INFO: green + fmt_str + reset,
                logging.WARNING: yellow + fmt_str + reset,
                logging.ERROR: red + fmt_str + reset,
                logging.CRITICAL: bold_red + fmt_str + reset
            }

        self.formatters = {
            level: logging.Formatter(fmt=f, datefmt=datefmt, style=style, validate=validate)
            for level, f in fmt.items()
        }

    def format(self, record):
        return self.formatters.get(record.levelno).format(record)


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
