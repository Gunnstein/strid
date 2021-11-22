# -*- coding: utf-8 -*-
import unittest
import numpy as np

from .._subspaceid import *
from .._subspaceid import create_block_hankel_matrix


class TestMisc(unittest.TestCase):
    def test_create_block_hankel(self):
        i = 2
        y = np.arange(16, dtype=float).reshape(2, -1) + 1
        j = y.shape[-1] - 2 * i + 1
        H = create_block_hankel_matrix(y, i)
        H_true = 1. / np.sqrt(j) * np.array([
            [ 1.,  2.,  3.,  4.,  5.],
            [ 9., 10., 11., 12., 13.],
            [ 2.,  3.,  4.,  5.,  6.],
            [10., 11., 12., 13., 14.],
            [ 3.,  4.,  5.,  6.,  7.],
            [11., 12., 13., 14., 15.],
            [ 4.,  5.,  6.,  7.,  8.],
            [12., 13., 14., 15., 16.]])
        np.testing.assert_almost_equal(H, H_true)


class TestStochasticSID:
    """Mixin this class to test stochastic sid
    """
    def set_data(self):
        np.random.seed(1)
        self.y = np.random.normal(size=(6, 20))
        self.fs = 1.
        self.ix_ref = [0, 1]

    def test_yref(self):
        np.testing.assert_almost_equal(
            self.sid.yref, self.y[self.ix_ref])

    def test_l(self):
        assert self.sid.l == self.y.shape[0]

    def test_r(self):
        assert self.sid.r == len(self.ix_ref)

    def test_s(self):
        assert self.sid.s == self.y.shape[1]

    def test_j(self):
        i = 2
        assert self.sid.j(i) == self.y.shape[1]-2*i+1


class TestCovarianceDrivenStochasticSID(unittest.TestCase,
                                        TestStochasticSID):
    def setUp(self):
        super().set_data()
        self.sid = CovarianceDrivenStochasticSID(
            self.y, self.fs, self.ix_ref)

    def test_T(self):
        i = 2

        T = self.sid._T(i)

        c0true = np.array(
            [ 0.46699503,  0.09508326, -0.44520017,  0.03928362, -0.09826325,
              0.43867734, -0.59208249,  0.05611628,  0.31816183, -0.30916625,
              0.08061237, -0.18516395])

        r0true = np.array([ 0.46699503,  0.02262047, -0.68611897, -0.0692071 ])

        np.testing.assert_almost_equal(T[:, 0], c0true)
        np.testing.assert_almost_equal(T[0], r0true)

    def test_svd_block_toeplitz(self):
        i = 2
        U, s, VH = self.sid._svd_block_toeplitz(i)

        s_true = np.array([1.55270166, 0.48958896, 0.36350019, 0.26182837])
        U0_true = np.array(
            [-0.52849248, -0.09563932,  0.29281458, -0.19050752,  0.14945724,
             -0.33584476,  0.48793835, -0.03620316, -0.34460092,  0.11897887,
             -0.11099778,  0.26413136])

        np.testing.assert_almost_equal(s, s_true)
        np.testing.assert_almost_equal(U[:, 0], U0_true)

    def test_perform(self):
        i = 2
        n = 4
        A, C, G, R0 = self.sid.perform(n, i)

        l = np.linalg.eigvals(A)
        ltrue = np.array( [-0.85856578+0.j, 0.41410653+1.02209135j,
                           0.41410653-1.02209135j, 0.40762358+0.j])
        ctrue = np.array([-0.658541  ,  0.19686591, -0.0528111 , -0.05481297])
        Gtrue = np.array([
            [ 0.91440777,  0.03810074],
            [-0.43195526, -0.2611433 ],
            [ 0.10086622, -0.44088996],
            [-0.11712863,  0.29171674]])
        R0true = np.array([
            [ 1.22735818, -0.03877381, -0.03550711,  0.07092257, -0.00224745,
              -0.03929179],
            [-0.03877381,  0.64959892, -0.01417519,  0.09136505,  0.0137499 ,
             0.246201  ],
            [-0.03550711, -0.01417519,  0.69818952,  0.0141953 ,  0.15913923,
             -0.2611224 ],
            [ 0.07092257,  0.09136505,  0.0141953 ,  1.08937737, -0.17276053,
              0.1664638 ],
            [-0.00224745,  0.0137499 ,  0.15913923, -0.17276053,  0.27133503,
             -0.13267658],
            [-0.03929179,  0.246201  , -0.2611224 ,  0.1664638 , -0.13267658,
             0.71467042]])

        np.testing.assert_almost_equal(l, ltrue)
        np.testing.assert_almost_equal(C[0], ctrue)
        np.testing.assert_almost_equal(G, Gtrue)
        np.testing.assert_almost_equal(R0, R0true)


class TestDataDrivenStochasticSID(unittest.TestCase,
                                        TestStochasticSID):
    def setUp(self):
        super().set_data()
        self.sid = DataDrivenStochasticSID(
            self.y, self.fs, self.ix_ref)

    def test_H(self):
        i = 2
        H = self.sid._Y(i)
        c0true = np.array(
            [0.39396162, -0.26693936, -0.14837272,  0.27763628, -0.12810047,
             0.21866787, -0.1812125 ,  0.12440375,  0.04524778,  0.0978611 ,
             -0.26023312,  0.12187278,  0.41048053, -0.07229813,  0.09945213,
             0.14396394])
        r0true = np.array(
            [0.39396162, -0.14837272, -0.12810047, -0.26023312,  0.20989218,
             -0.55820513,  0.42317901, -0.18461979,  0.07737835, -0.0604812 ,
             0.35461326, -0.49965751, -0.07819766, -0.09314686,  0.27497948,
             -0.26676282, -0.04181998])
        np.testing.assert_almost_equal(H[:, 0], c0true)
        np.testing.assert_almost_equal(H[0], r0true)

    def test_LQ_decomposition_block_hankel(self):
        i = 4
        L, Q = self.sid._LQ_decomposition_block_hankel(i)
        self.assertEqual(L.shape, (32, 13))
        l = L.ravel()
        l = l[l!=0.][11:44]
        l_true = np.array(
            [0.03629694, -0.57803404,  0.21057971, -0.96271933, -0.20831074,
             -0.15551718,  0.03306401, -0.2976252 , -0.09848928,  0.53282752,
             0.54716481,  0.07040023,  0.24188369,  0.06869012,  0.62088876,
             0.14667863, -0.8789843 , -0.13467873, -0.04278225,  0.26440878,
             -0.21188353, -0.04721381,  0.07061954, -0.20134536,  0.45366388,
             -0.70913006,  0.00666429, -0.07021026,  0.12267643, -0.20182931,
             -0.08560008,  0.5011539 ,  0.00213558])
        np.testing.assert_almost_equal(l, l_true)

    def test_svd_weighted_projection(self):
        i = 3
        U, s = self.sid._svd_weighted_projection(i)
        s_true = np.array(
            [1.54593941, 1.03892179, 0.98797918, 0.70407908, 0.56286593,
             0.43802298])
        U0_true = np.array(
            [-0.43667576, -0.32677184, -0.1400151 ,  0.16850738, -0.21262854,
             -0.12236128,  0.43174673,  0.21134349, -0.19689495,  0.00157889,
             -0.02880627,  0.28121598, -0.39106887,  0.10111087, -0.22320098,
             -0.0640684 ,  0.00851684, -0.16566352])
        np.testing.assert_almost_equal(s, s_true)
        np.testing.assert_almost_equal(U[0], U0_true)

    def test_perform(self):
        i = 2
        n = 4
        A, C, G, R0 = self.sid.perform(n, i)

        l = np.linalg.eigvals(A)

        ltrue = np.array(
            [2.40986464+0.j, -0.95082552+0.j,
             0.53414245+0.46955631j, 0.53414245-0.46955631j])
        ctrue = np.array([-0.5730579 , 0.17945471, -0.03134651,
                          -0.09653276])
        Gtrue = np.array([
            [ 0.06524138, 0.0038382 ],
            [-0.01500376, -0.01277695],
            [ 0.00812859, -0.02593309],
            [ 0.00026472, 0.00405582]])
        R0true = np.array([
            [ 0.07333822,  0.0044295 , -0.00504298,  0.01026364, -0.00073858,
              0.00156139],
            [ 0.0044295 ,  0.03432385, -0.00046839, -0.00346933, -0.00089224,
              0.00943514],
            [-0.00504298, -0.00046839,  0.04275304,  0.00199578,  0.008111  ,
             -0.01585899],
            [ 0.01026364, -0.00346933,  0.00199578,  0.06597956, -0.0135063 ,
              0.00396619],
            [-0.00073858, -0.00089224,  0.008111  , -0.0135063 ,  0.01678103,
             -0.00966276],
            [ 0.00156139,  0.00943514, -0.01585899,  0.00396619, -0.00966276,
              0.04299954]])

        np.testing.assert_almost_equal(l, ltrue)
        np.testing.assert_almost_equal(C[0], ctrue)
        np.testing.assert_almost_equal(G, Gtrue)
        np.testing.assert_almost_equal(R0, R0true)

class TestCombinedDeterministicStochasticSID(unittest.TestCase,
                                             TestStochasticSID):
    def setUp(self):
        super().set_data()
        self.u = np.random.normal(size=(3, self.y.shape[1]))
        self.sid = CombinedDeterministicStochasticSID(
            self.u, self.y, self.fs, self.ix_ref)

    def test_m(self):
        assert self.sid.m == self.u.shape[0]

    def test_U(self):
        i = 2
        U = self.sid._U(i)

        Ur = np.array(
            [-0.00597049, -0.18800431,  0.30893119,  0.47709225, -0.45062679,
             0.29981382,  0.39476329,  0.08197988, -0.29086522,  0.209392  ,
             -0.04387962, -0.14647227, -0.29833292,  0.13352496,  0.19228391,
             -0.15122842,  0.12625831])
        Uc = np.array(
            [-0.00597049, -0.04524982, -0.12541383, -0.18800431, -0.024677  ,
             -0.24181452,  0.30893119,  0.21073585,  0.06034266,  0.47709225,
             0.18200156, -0.07194605])

        np.testing.assert_almost_equal(Ur, U[0])
        np.testing.assert_almost_equal(Uc, U[:, 0])

    def test_LQ_decomposition_block_hankel(self):
        i = 2
        L = self.sid._LQ_decomposition_block_hankel(i)
        l = L.flatten()
        l = l[l!=0.]
        pick = [1, 9, 23, 33, 49, 82]
        l = l[pick]
        ltrue = np.array([ 0.23565322, 0.9855933 , -0.0019167 ,
                           -0.4457208 , -0.09636363, -0.54876988])
        np.testing.assert_almost_equal(l, ltrue)

    def test_svd_weighted_projection(self):
        i = 2
        U, s = self.sid._svd_weighted_projection(i)

        strue = np.array([1.25307387e+00, 1.01255905e+00,
                          8.71553828e-01, 7.61318658e-01,
                          6.93714114e-01, 4.23724381e-01,
                          3.58667462e-01, 2.97083432e-01,
                          1.99564962e-01, 9.55807268e-02,
                          2.06170471e-16, 5.05450707e-17])
        U0true = np.array( [0.46248273, -0.07687227, -0.15602234,
                            -0.25099892, 0.12367738, -0.01755581, -0.60222401,
                            -0.35909031, 0.12426549, -0.30323888, 0.035321 ,
                            -0.27747023])

        np.testing.assert_almost_equal(s, strue)
        np.testing.assert_almost_equal(U[:, 0], U0true)

    def test_weighted_projection(self):
        i = 2
        O = self.sid._weighted_projection(i)
        Oc0 = np.array([-0.50804223, 0.01983242, 0.18091674,
                        0.02943713, 0.0822605 , -0.01105496,
                        0.39117671, 0.03190461, -0.09970498,
                        0.07370108, 0.11085066, 0.00497335])

        Or0 = np.array([-0.50804223, -0.18962527, -0.0527331 ,
                        0.27852579, -0.13006793, 0.03863123, -0.008255
                        , 0.00353266, 0.01890065, -0.24563044,
                        -0.28942989, -0.04965905, 0.00518684,
                        0.07851982, -0.1294491 , 0.30120054])

        np.testing.assert_almost_equal(O[:, 0], Oc0)
        np.testing.assert_almost_equal(O[0], Or0)


if __name__ == "__main__":
    unittest.main()
