"""Implements different models."""

import abc
import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.interpolate import splrep, splev
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeRegressor

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


class WeakLearner(abc.ABC):
    """Weak learner to be used in MLSGD algorithm.

    This is a wrapper class to provide an uniform interface for different kinds
    of learners.

    """
    def __init__(self):
        pass


    @abc.abstractmethod
    def fit(self, X, y):
        """Fit weak learner to gradient estimate."""


    @abc.abstracmethod
    def __call__(self, X):
        """Call weak estimator on some input."""


class Tree(WeakLearner):
    """ We use the sklearn defaults (which include L2 loss) and only change the
    max depth of the tree.

    """
    def __init__(self, max_depth: int = 5):
        super().__init__()
        self.estimator = DecisionTreeRegressor(max_depth=max_depth)


    def fit(self, X: np.ndarray, y: np.ndarray):
        self.estimator.fit(X, y)


    def __call__(self, X: np.ndarray):
        return self.estimator.predict(X)


class Spline(WeakLearner):
    """ We use the sklearn defaults (which include L2 loss) and only change the
    max depth of the tree.

    """
    def __init__(self, degree: int = 3):
        super().__init__()
        self.degree = degree


    def fit(self, X: np.ndarray, y: np.ndarray):
        self.estimator = splrep(X, y, k=self.degree)


    def __call__(self, X: np.ndarray):
        return splev(X, self.estimator)


@dataclass
class MLSGDDeconvolution1D(BaseEstimator):
    """Implements a gradient-boost like algorithm for the deconvolution
    problem in 1D.

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
    weak_learner_factory : function which return WeakLearner
        A factory which creates a weak learner for each iteration of the
        algorithm.
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
    weak_learner_factory: Callable[[], WeakLearner] = lambda: Spline()
    learning_rate: Callable[[int], float] = lambda i: 1/np.sqrt(i)
    loss: losses.LossWithGradient = losses.SquaredLoss()
    fixed_learning_rate: bool = False


    def phi(self, x: np.ndarray, w: np.ndarray) -> np.ndarray:
        return self.kernel(x - w)


    def fit(self, X: np.ndarray, y: np.ndarray) -> BaseEstimator:
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
