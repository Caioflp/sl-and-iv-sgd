"""Logging related utilities."""

import logging


FORMATTER = '%(asctime)s - %(name)s [%(levelname)s] %(message)s'


def setup_main_logger() -> logging.Logger:
    """Creates the main logger.

    Concentrates boilerplate code to setup logger and handler.

    """
    logger = logging.getLogger("main")
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter(FORMATTER)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger
