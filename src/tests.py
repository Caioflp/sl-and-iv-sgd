""" Tests for some pieces of our codebase """

import unittest

import numpy as np

from src.utils.math import convolve


class TestConvolution(unittest.TestCase):
    """Tests convolution function."""
    def test_null_function(self):
        domain = np.linspace(0, np.pi, 50)
        self.assertEqual(
            convolve(lambda x: 0, lambda x: 0, 0.5, domain), 0
        )


    def test_constant_function(self):
        domain = np.linspace(0, np.pi, 50)
        self.assertAlmostEqual(
            convolve(np.sin, lambda x: 1, 0.5, domain),
            2.0,
            places=4,
        )


    def test_specific_example(self):
        domain = np.linspace(-2, 3, 1000)
        # Taken from https://lpsa.swarthmore.edu/Convolution/Convolution2.html
        def func(t):
            return np.exp(-2*t) * (t > 0)
        def kernel(t):
            return 0 <= t <= 1
        inputs = [-1,
                  0.5,
                  2]
        outputs = [
            0,
            0.5 * (1 - np.exp(-2*inputs[1])),
            np.exp(-2 * (2 - 1)) * 0.5 * (1 - np.exp(-2))
        ]
        for t, ans in zip(inputs, outputs):
            self.assertAlmostEqual(
                convolve(func, kernel, t, domain),
                ans,
                places=3,
            )


if __name__ == "__main__":
    unittest.main()
