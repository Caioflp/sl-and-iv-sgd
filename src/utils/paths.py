"""Path related utilities"""

import logging
import os
from contextlib import AbstractContextManager
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
module_logger = logging.getLogger('experiment.utils')


class InPath(AbstractContextManager):
    """Context manager to run code with a given path as working dir."""
    def __init__(self, path: Path):
        self.origin = os.getcwd()
        self.destiny = path
        self.logger = logging.getLogger('experiment.utils.InPath')
        self.logger.debug(f'Current working dir: {self.origin}.')

    def __enter__(self) -> None:
        self.logger.info(f'Entering experiment path {self.destiny}.')
        os.chdir(self.destiny)
        self.logger.info(f'Current working dir: {self.origin}.')

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.logger.info('Exiting experiment path.')
        os.chdir(self.origin)
        self.logger.info(f'Current working dir: {self.origin}.')
