# -*- coding: utf-8 -*-
import numpy as np
import unittest

from ..utils import ShearFrame
from .. import *


class TestShearFrame(unittest.TestCase):
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


class TestUtils(unittest.TestCase):
    def test_find_rayleigh_damping_coeffs(self):
        ws = np.array([8., 22.])
        xis = np.array([.1, .24])
        a, b = find_rayleigh_damping_coeffs(ws, xis)
        xis_approx = .5*(a/ws + b*ws)
        np.testing.assert_almost_equal(xis, xis_approx)

    def test_w2f(self):
        assert w2f(np.pi) == .5

    def test_f2w(self):
       assert f2w(-1) == -2*np.pi

    def test_modal_scale_factor(self):
        u = np.array([1., -0.2])
        v = np.pi*u
        assert modal_scale_factor(u, v) == np.pi

    def test_modal_phase_collinearity(self):
        u1 = np.ones(10)*np.exp(1j*np.pi/4)
        u0 = u1*np.exp(1j*np.linspace(0, 2*np.pi, u1.size))
        for v, u in [(1., u1), (0.01, u0)]:
            self.assertAlmostEqual(modal_phase_collinearity(u), v, places=2)

    def test_mean_phase(self):
        mp = np.pi/4
        u1 = np.ones(10)*np.exp(1j*mp)
        self.assertAlmostEqual(mean_phase(u1), mp)

    def test_mean_phase_deviation(self):
        n = 10
        phases = np.array([1.]*5 + [-1.]*5)*np.pi/8
        u = np.ones(n) * np.exp(1.j*phases)
        self.assertAlmostEqual(mean_phase_deviation(u), np.pi/8)

    def test_norm2(self):
        assert norm2([1., 2., 1.]) == np.sqrt(6)

    def test_get_frequency_vector(self):
        n = 10
        fs = 8
        f = get_frequency_vector(fs, n)

        assert f[-1] == fs/2
        assert f[0] == 0.
        assert f.size == n

    def test_modal_assurance_criterion(self):
        u = np.array([7., 0., 0.])
        v = np.array([0., 1.+3j, 0.])


        assert modal_assurance_criterion(u, u) == 1.
        assert modal_assurance_criterion(u, -u) == 1.

        assert modal_assurance_criterion(v, v) == 1.
        assert modal_assurance_criterion(v, -v) == 1.

        assert modal_assurance_criterion(u, v) == 0.


        self.assertAlmostEqual(
            modal_assurance_criterion(u, u+v),
            (7*7)**2 / (7**2*norm2(u+v)**2))




if __name__ == "__main__":
    unittest.main()
