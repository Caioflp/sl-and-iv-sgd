"""Utilities"""

import logging
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.integrate import simpson


logger = logging.getLogger("main.math")


def convolve_array(
    func_samples: np.ndarray,
    kernel: Callable[[np.ndarray], np.ndarray],
    x: float,
    domain: np.ndarray,
) -> float:
    """
    Compute the convolution of `kernel` and a function at `x`, using Simpson's
    rule with samples computed at `domain`.
    The function samples are already given in `array`.

    """
    assert domain[0] <= x <= domain[-1]
    # samples = np.vectorize(lambda w: kernel(x - w))(domain) * func_samples
    samples = kernel(x - domain) * func_samples
    return simpson(samples, domain)


def convolve(
    func: Callable[[np.ndarray], np.ndarray],
    kernel: Callable[[np.ndarray], np.ndarray],
    x: float,
    domain: np.ndarray,
) -> float:
    """
    Compute the convolution of `kernel` and `func` at `x` using Simpson's rule
    with samples computed at `domain`.

    """
    assert domain[0] <= x <= domain[-1]
    # samples = np.vectorize(lambda w: kernel(x - w) * func(w))(domain)
    samples = kernel(x - domain) * func(domain)
    return simpson(samples, domain)
