"""Implements different models."""

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
import sklearn

from src import losses
from src import utils


logger = logging.getLogger("main.experiment.model")


@dataclass
class SGDSIPDeconvolution1D(sklearn.base.BaseEstimator):
    """Implements a SGD algorithm for the deconvolution problem in 1D

    Parameters
    ----------
    start : float
        Supremum of domain interval.
    end : float
        Infimum of domain interval.
    step : float
        Discretization step for domain.
    kernel : function
        Convolution kernel.
    initial_guess : function
        Initial guess for target function.
    learning_rate : function
        Function which returns the learning rate for iteration `i`, where `i`
        starts at 1.
    loss : losses.SquaredLoss
        Loss function to be used.
    fixed_learning_rate: bool
        Controls whether to use `1/np.sqrt(n_iter)` as learning rate.

    """
    start: float = -10
    end: float = 10
    step: float = 1E-2
    kernel: Callable[[float], float] = lambda x: np.float64(x > 0)
    initial_guess: Callable[[float], float] = lambda x: 0.0
    learning_rate: Callable[[int], float] = lambda i: 1/np.sqrt(i)
    loss: losses.LossWithGradient = losses.SquaredLoss()
    fixed_learning_rate: bool = False


    def phi(self, x: np.ndarray, w: np.ndarray) -> np.ndarray:
        return self.kernel(x - w)


    def fit(self, X: np.ndarray, y: np.ndarray) -> sklearn.base.BaseEstimator:
        """Fit model.

        Parameters
        ----------
        X: np.ndarray of shape (n_samples,)
        y: np.ndarray of shape (n_samples,)

        """
        n_points = int((self.end-self.start)/self.step+1)
        # pylint: disable=attribute-defined-outside-init
        self.discretized_domain = np.linspace(
            self.start,
            self.end,
            n_points,
        )
        initial_guess_array = \
                np.vectorize(self.initial_guess)(self.discretized_domain)
        n_iter = X.shape[0]
        if self.fixed_learning_rate:
            # pylint: disable=unused-argument
            def learning_rate(x: float):
                return np.float64(1/np.sqrt(n_iter))
        else:
            learning_rate = self.learning_rate
        self.estimates_ = np.empty((n_iter+1, n_points), dtype=np.float64)
        self.estimates_[0, :] = initial_guess_array
        for it, (sample_x, sample_y) in enumerate(zip(X, y), start=1):
            logger.info(f"Iteration {it} of {n_iter}.")
            current_estimate_convolution = utils.math.convolve_array(
                self.estimates_[it-1, :],
                self.kernel,
                sample_x,
                self.discretized_domain,
            )
            grad_estimate = (
                self.phi(sample_x, self.discretized_domain)
                * self.loss.grad(sample_y, current_estimate_convolution)
            )
            self.estimates_[it, :] = (self.estimates_[it-1, :]
                                      - learning_rate(n_iter) * grad_estimate)
        self.estimate_ = 1/n_iter * np.sum(self.estimates_, axis=0)

        return self
