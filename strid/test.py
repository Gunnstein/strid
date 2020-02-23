# -*- coding: utf-8 -*-
import numpy as np
import unittest

from .utils import ShearFrame, find_rayleigh_damping_coeffs
from ._integration import GeneralizedSystem
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


class TestIntegrator(unittest.TestCase):
    def setUp(self):
        self.system = ShearFrame(5, 1e3, 1e6)
        M, C, K = self.system.M, self.system.C, self.system.K
        self.generalized_system = GeneralizedSystem(M, C, K)

        # F = np.zeros((M.shape[0], 1000)) + 100000
        F = np.random.normal(size=(M.shape[0], 1000)) * 100000 + 24000
        dt = 1/500.
        v0 = np.zeros(M.shape[0])
        d0 = self.system.find_mode_shape(1)
        A, V, D = generalized_alpha_method(M, C, K, F, dt, d0, v0)
        self.A = A
        self.V = V
        self.D = D

    def test_Kdyn(self):
        true = np.array(
            [[ 5500., -2500.,     0.,     0.,     0.,],
             [-2500.,  5500., -2500.,     0.,     0.,],
             [    0., -2500.,  5500., -2500.,     0.,],
             [    0.,     0., -2500.,  5500., -2500.,],
             [    0.,     0.,     0., -2500.,  3000.,],])
        np.testing.assert_almost_equal(self.generalized_system.Kdyn(.1), true)

    def test_Mf(self):
        true = np.eye(5)*500.
        np.testing.assert_equal(self.generalized_system.Mf(.1), true)

    def test_Cf(self):
        true = np.array(
            [[100000., -50000.,      0.,      0.,      0.,],
             [-50000., 100000., -50000.,      0.,      0.,],
             [     0., -50000., 100000., -50000.,      0.,],
             [     0.,      0., -50000., 100000., -50000.,],
             [     0.,      0.,      0., -50000.,  50000.,],])
        np.testing.assert_equal(self.generalized_system.Cf(.1), true)

    def test_Kf(self):
        np.testing.assert_equal(self.generalized_system.Kf(.1), self.system.K)

    def test_D(self):
        true = self.system.find_mode_shape(1)
        np.testing.assert_equal(true, self.D[:, 0])
        true = np.array([0.06143933, 0.14867399, 0.16715818,
                         0.26068709, 0.20816236])
        np.testing.assert_almost_equal(true, self.D[:, 800], decimal=6)

    def test_V(self):
        true = np.zeros(5)
        np.testing.assert_equal(true, self.V[:, 0])

if __name__ == "__main__":
    unittest.main()
