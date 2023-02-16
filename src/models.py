"""Implements different models."""

import abc
import logging
from typing import Callable

import numpy as np
from scipy.interpolate import splrep, splev
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeRegressor

from src import losses
from src import utils


logger = logging.getLogger("main.experiment.model")


class SGDSIPDeconvolution1D(BaseEstimator):
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
    def __init__(
        self,
        start: float = -10,
        end: float = 10,
        step: float = 1E-2,
        kernel: Callable[[float], float] = lambda x: np.float64(x > 0),
        initial_guess: Callable[[float], float] = lambda x: 0.0,
        learning_rate: Callable[[int], float] = lambda i: 1/np.sqrt(i),
        loss: losses.LossWithGradient = losses.SquaredLoss(),
        fixed_learning_rate: bool = False,
        name: str = "SGD",
    ) -> None:
        self.start = start
        self.end = end
        self.step = step
        self.kernel = kernel
        self.initial_guess = initial_guess
        self.learning_rate = learning_rate
        self.loss = loss
        self.fixed_learning_rate = fixed_learning_rate
        self.name = name

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
            self.learning_rate = learning_rate
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
            self.estimates_[it, :] = (
                self.estimates_[it-1, :]
                - self.learning_rate(it) * grad_estimate
            )
        self.estimate_ = 1/n_iter * np.sum(self.estimates_, axis=0)

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray):
        """ Predict method for compatibility with other estimators. """
        assert self.is_fitted
        assert X.shape == self.estimate_.shape
        return self.estimate_


class NAGSIPDeconvolution1D(BaseEstimator):
    """Implements a Nesterov Accelerated Gradient (NAG) type algorithm
    for the deconvolution problem in 1D.

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
    def __init__(
        self,
        start: float = -10,
        end: float = 10,
        step: float = 1E-2,
        kernel: Callable[[float], float] = lambda x: np.float64(x > 0),
        initial_guess: Callable[[float], float] = lambda x: 0.0,
        loss: losses.LossWithGradient = losses.SquaredLoss(),
        learning_rate: float = 1E-2,
        alpha: float = 3,
        name: str = "SGD",
    ) -> None:
        self.start = start
        self.end = end
        self.step = step
        self.kernel = kernel
        self.initial_guess = initial_guess
        self.loss = loss
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.name = name

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
        # x_{ -1 } and x_0 are required by NAG, hence the `+2`
        self.estimates_ = np.empty((n_iter+2, n_points), dtype=np.float64)
        self.estimates_[0, :] = initial_guess_array
        self.estimates_[1, :] = initial_guess_array
        for it, (sample_x, sample_y) in enumerate(zip(X, y), start=1):
            logger.info(f"Iteration {it} of {n_iter}.")
            previous_estimate = self.estimates_[it-1]
            current_estimate = self.estimates_[it]
            last_step = current_estimate - previous_estimate
            weight = (it - 1) / (it + self.alpha - 1)
            accelerated_estimate = (
                current_estimate + weight * last_step
            )
            accelerated_estimate_conv = utils.math.convolve_array(
                accelerated_estimate,
                self.kernel,
                sample_x,
                self.discretized_domain,
            )
            grad_estimate = (
                self.phi(sample_x, self.discretized_domain)
                * self.loss.grad(sample_y, accelerated_estimate_conv)
            )
            self.estimates_[it+1, :] = (
                accelerated_estimate
                - self.learning_rate * grad_estimate
            )
        self.estimate_ = 1/n_iter * np.sum(self.estimates_, axis=0)

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray):
        """ Predict method for compatibility with other estimators. """
        assert self.is_fitted
        assert X.shape == self.estimate_.shape
        return self.estimate_


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

    @abc.abstractmethod
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
        X = np.array(X).reshape(-1, 1)
        self.estimator.fit(X, y)

    def __call__(self, X: np.ndarray):
        X = np.array(X)
        if len(X.shape) == 0:
            X = X.reshape(1, 1)
            return self.estimator.predict(X).item()
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
            return self.estimator.predict(X)
        return self.estimator.predict(X)


class Spline(WeakLearner):
    """ We use the scipy implementation of B-splines. """
    def __init__(self, degree: int = 3):
        super().__init__()
        self.degree = degree

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.estimator = splrep(X, y, k=self.degree)

    def __call__(self, X: np.ndarray):
        return splev(X, self.estimator)


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
    def __init__(
        self,
        start: float = -10,
        end: float = 10,
        step: float = 1E-2,
        kernel: Callable[[float], float] = lambda x: np.float64(x > 0),
        initial_guess: Callable[[float], float] = lambda x: 0.0,
        weak_learner_factory: Callable[[], WeakLearner] = lambda: Spline(),
        learning_rate: Callable[[int], float] = lambda i: 1/np.sqrt(i),
        loss: losses.LossWithGradient = losses.SquaredLoss(),
        fixed_learning_rate: bool = False,
        name: str = "MLSGD",
    ) -> None:
        self.start = start
        self.end = end
        self.step = step
        self.kernel = kernel
        self.initial_guess = initial_guess
        self.weak_learner_factory = weak_learner_factory
        self.learning_rate = learning_rate
        self.loss = loss
        self.fixed_learning_rate = fixed_learning_rate
        self.name = name

    def predict_for_fit(self, X: float):
        """ Apply estimators to obtain prediction during training. """
        assert hasattr(self, "estimator_list")
        result = self.initial_guess(X)
        for i, estimator in enumerate(self.estimator_list, start=1):
            result -= self.learning_rate(i) * estimator(X)
        return result

    def predict(self, X: float):
        """ Apply estimators to obtain final prediction.

        What we are doing is equivalent to 1/n sum_i=1^n g_i,
        where g_i = g_i-1 - alpha_i * h_i, with g_0 being the initial guess,
        alpha_i the learning rate for iteration i and h_i the weak learner for
        iteration i.

        We basically expandend the sum in terms of alpha_i and h_i and
        collected like terms.

        """
        N = len(self.estimator_list)
        result = (N + 1) * self.initial_guess(X)
        for i, estimator in enumerate(self.estimator_list, start=1):
            result -= (N + 1 - i) * self.learning_rate(i) * estimator(X)
        result /= N
        return result

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
        self.discretized_domain = np.linspace(
            self.start,
            self.end,
            n_points,
        )

        self.estimator_list = []

        n_iter = X.shape[0]
        if self.fixed_learning_rate:
            # pylint: disable=unused-argument
            def learning_rate(x: float):
                return np.float64(1/np.sqrt(n_iter))
            self.learning_rate = learning_rate

        for it, (sample_x, sample_y) in enumerate(zip(X, y), start=1):
            logger.info(f"Iteration {it} of {n_iter}.")
            current_estimate_convolution = utils.math.convolve(
                self.predict_for_fit,
                self.kernel,
                sample_x,
                self.discretized_domain,
            )
            grad_estimate = (
                self.phi(sample_x, self.discretized_domain)
                * self.loss.grad(sample_y, current_estimate_convolution)
            )
            new_estimator = self.weak_learner_factory()
            new_estimator.fit(self.discretized_domain, grad_estimate)
            self.estimator_list.append(new_estimator)

        return self


class MLNAGDeconvolution1D(BaseEstimator):
    """Implements a Nesterov Accelerated Gradient (NAG) type algorithm
    for the deconvolution problem in 1D.

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
    def __init__(
        self,
        start: float = -10,
        end: float = 10,
        step: float = 1E-2,
        kernel: Callable[[float], float] = lambda x: np.float64(x > 0),
        initial_guess: Callable[[float], float] = lambda x: 0.0,
        weak_learner_factory: Callable[[], WeakLearner] = lambda: Spline(),
        loss: losses.LossWithGradient = losses.SquaredLoss(),
        learning_rate: float = 1E-2,
        alpha: float = 3,
        name: str = "MLNAG",
    ) -> None:
        self.start = start
        self.end = end
        self.step = step
        self.kernel = kernel
        self.initial_guess = initial_guess
        self.weak_learner_factory = weak_learner_factory
        self.loss = loss
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.name = name

    def predict_for_fit(self, X: float):
        """ Apply estimators to obtain prediction during training. """
        assert hasattr(self, "estimator_list")
        result = self.initial_guess(X)
        for i, estimator in enumerate(self.estimator_list, start=1):
            result -= self.learning_rate(i) * estimator(X)
        return result

    def predict(self, X: float):
        """ Apply estimators to obtain final prediction.

        What we are doing is equivalent to 1/n sum_i=1^n g_i,
        where g_i = g_i-1 - alpha_i * h_i, with g_0 being the initial guess,
        alpha_i the learning rate for iteration i and h_i the weak learner for
        iteration i.

        We basically expandend the sum in terms of alpha_i and h_i and
        collected like terms.

        """
        N = len(self.estimator_list)
        result = (N + 1) * self.initial_guess(X)
        for i, estimator in enumerate(self.estimator_list, start=1):
            result -= (N + 1 - i) * self.learning_rate(i) * estimator(X)
        result /= N
        return result

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
        self.discretized_domain = np.linspace(
            self.start,
            self.end,
            n_points,
        )
        self.estimator_list = []
        n_iter = X.shape[0]
        # x_{ -1 } and x_0 are required by NAG, hence the `+2`
        self.estimates_ = np.empty((n_iter+2, n_points), dtype=np.float64)
        self.estimates_[0, :] = initial_guess_array
        self.estimates_[1, :] = initial_guess_array
        for it, (sample_x, sample_y) in enumerate(zip(X, y), start=1):
            logger.info(f"Iteration {it} of {n_iter}.")
            previous_estimate = self.estimates_[it-1]
            current_estimate = self.estimates_[it]
            last_step = current_estimate - previous_estimate
            weight = (it - 1) / (it + self.alpha - 1)
            accelerated_estimate = (
                current_estimate + weight * last_step
            )
            accelerated_estimate_conv = utils.math.convolve_array(
                accelerated_estimate,
                self.kernel,
                sample_x,
                self.discretized_domain,
            )
            grad_estimate = (
                self.phi(sample_x, self.discretized_domain)
                * self.loss.grad(sample_y, accelerated_estimate_conv)
            )
            self.estimates_[it+1, :] = (
                accelerated_estimate
                - self.learning_rate * grad_estimate
            )
        self.estimate_ = 1/n_iter * np.sum(self.estimates_, axis=0)

        self.is_fitted = True
        return self
