"""Implements loss functions and their gradients.

Those functions are intended to be used with SGDSIP and MLSGD estimators.

"""

import abc
import logging

import numpy as np


logger = logging.getLogger("main.experiment.model.loss")


class LossWithGradient(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        pass


    @abc.abstractmethod
    def grad(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute gradient w.r.t. `y_pred`."""


class SquaredLoss(LossWithGradient):
    """Usual sum of squares loss."""
    def __init__(self):
        pass

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return 0.5 * np.sum(np.square(y_true - y_pred))


    def grad(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        # logger.debug(f"Gradient: {y_pred - y_true}")
        return y_pred - y_true
