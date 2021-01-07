# -*- coding: utf-8 -*-
import numpy as np
import unittest

from .utils import ShearFrame, find_rayleigh_damping_coeffs
from . import *

np.random.seed(2)


class TestUtils(unittest.TestCase):
    def setUp(self):
        self.n = 5
        self.m = 1e3
        self.k = 1e6
        self.shear_frame = ShearFrame(self.n, self.m, self.k)

    def get_natural_frequencies(self):
        k, m, n = self.k, self.m, self.n
        freqs = np.array([
            2 * np.sqrt(k / m) * np.sin(np.pi / 2 * (2*r-1) / (2*n+1))
            for r in range(1, self.n+1)])
        return freqs

    def get_mode_shapes(self):
        Q = []
        for r in range(1, self.n+1):
            q = np.array([np.sin(i*np.pi*(2*r-1)/(2*self.n+1))
                      for i in range(1, self.n+1)])
            q /= np.linalg.norm(q, 2)
            Q.append(q)
        Q = np.array(Q).T
        return Q

    def test_sf_k(self):
        assert self.k == self.shear_frame.k

    def test_sf_m(self):
        assert self.m == self.shear_frame.m

    def test_sf_n(self):
        assert self.n == self.shear_frame.n

    def test_eigvals(self):
        fn_true = self.get_natural_frequencies()
        M, K = self.shear_frame.M, self.shear_frame.K
        l, Q = np.linalg.eig(np.linalg.solve(M, K))
        fn = np.sqrt(l)
        fn.sort()
        np.testing.assert_almost_equal(fn_true, fn)

    def test_eigvecs(self):
        M, K = self.shear_frame.M, self.shear_frame.K
        l, Q = np.linalg.eig(np.linalg.solve(M, K))
        n = np.argsort(l)
        Q = Q[:, n]
        Q_true = self.get_mode_shapes()
        np.testing.assert_almost_equal(np.abs(Q), np.abs(Q_true))

    def test_find_rayleigh_damping_coeffs(self):
        ws = np.array([8., 22.])
        xis = np.array([.1, .24])
        a, b = find_rayleigh_damping_coeffs(ws, xis)
        xis_approx = .5*(a/ws + b*ws)
        np.testing.assert_almost_equal(xis, xis_approx)


if __name__ == "__main__":
    unittest.main()
