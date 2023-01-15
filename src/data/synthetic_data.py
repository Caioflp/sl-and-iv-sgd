""" Functions for generating synthetic data """

import logging
from typing import Callable, Tuple

import numpy as np

from src.utils.math import convolve


logger = logging.getLogger("main.experiment.data")
RNG = np.random.default_rng()


def make_noised_convolution(
    start: float = -10,
    end: float = 10,
    func: Callable[[float], float] = lambda w: np.exp(-w**2),
    kernel: Callable[[float], float] = lambda z: np.float64(z > 0),
    step_gen: float = 1E-2,
    step_obs: float = 1E-1,
    noise_var: float = 2,
    random_observations: bool = False,
    n_samples: int = 201,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convolve `kernel` and `func` in [`start`, `end`] and add noise.

    Computes approximate convolution of `kernel` and `func` in the closed
    interval [`start`, `end`] using Simpson's rule. Then, adds i.i.d. Gaussian
    noise with zero mean and `noise_var` variance.
    Data is generated using `step_gen` discretization step size, but is returned
    with `step_obs` discretization step size.

    If `random_observations` is set to `True`, then `step_obs` is ignored and a
    total of `n_samples` values of x will be randomly drawn from a uniform
    distribution in [`start`, `end`].

    Returns
    -------
    np.ndarray
        Points where convolution was computed.
    np.ndarray
        Convolution with noise.

    """
    step_ratio = step_obs / step_gen
    assert step_ratio.is_integer()
    fine_domain = np.linspace(start, end, num=int((end-start)/step_gen+1))
    if random_observations:
        coarse_domain = RNG.uniform(low=start, high=end, size=n_samples)
    else:
        coarse_domain = np.linspace(start, end, num=int((end-start)/step_obs+1))
    conv_func = np.vectorize(
        lambda x: convolve(func, kernel, x, fine_domain)
    )
    convolution = conv_func(coarse_domain)
    # logger.debug(convolution)
    noised_convolution = (convolution
                          + RNG.normal(loc=0,
                                       scale=np.sqrt(noise_var),
                                       size=convolution.shape)
                         )
    return coarse_domain, noised_convolution, convolution
