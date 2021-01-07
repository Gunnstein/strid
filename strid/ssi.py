# -*- coding:utf-8 -*-
"""
Implementation of stochastic subspace identification methods

The implementation and terminology is based on the following resources

    `Rainieri, C., Fabbrocino, G., 2014. Operational Modal Analysis of
    Civil Engineering Structures. Springer.
    https://doi.org/10.1007/978-1-4939-0767-0`
"""
import abc
import functools
import logging
import numpy as np
from . import utils


logger = logging.getLogger(__name__)


__all__ = ["CovSSI", "StateSpaceSystem",]


class CovSSI(abc.ABC):
    def __init__(self, Y, fs):
        """Define a covariance driven stochastic subspace identification object.

        Arguments
        ---------
        Y : ndarray
            Data (measurement) array where each row corresponds to sensor output
            and each column corresponds to the sensor output at a particular
            time.
        fs : float
            Sampling frequency (Hz).
        """
        self._Y = Y
        self.fs = fs

    @property
    def Y(self):
        return self._Y

    @Y.setter
    def Y(self, value):
        raise AttributeError("Cannot change the measurement array after initialization, create a new instance.")

    def psd(self, **kw):
        """Compute power spectral density matrix with Welch's method

        Arguments
        ---------
        kw : dict
            See keywords to scipy.signal.csd and scipy.signal.welch.

        Returns
        -------
        f : 1darray
            Frequency vector of the psd matrix
        Pyy : 3darray
            PSD for the system, trace of the psd matrix.
        """
        psd_matrix = utils.find_psd_matrix(self.Y, fs=self.fs, **kw)
        Pyy = np.trace(psd_matrix, axis1=1, axis2=2)
        f = utils.get_frequency_vector(self.fs, Pyy.size)
        return f, Pyy.real

    def _R(self, i):
        """Correlation matrix of data array

        Correlation matrix between data array with `i` last samples
        removed and data array with first `i` samples removed.

        Arguments
        ---------
        i : int
            Time lag / shift

        Returns
        -------
        ndarray
            Correlation matrix
        """
        l, N = self.Y.shape
        i = np.abs(i)
        return self.Y[:, :N-i].dot(self.Y[:, i:].T) /(N-i)

    def _H(self, block_rows):
        """Block hankel matrix from data array

        Arguments
        ---------
        block_rows : int
            Number of block rows

        Returns
        -------
        ndarray
            Block hankel matrix from data array
        """
        l, N = self.Y.shape
        i = block_rows
        j = N - 2*i + 1
        H = np.zeros((l*2*i, j))
        for m in range(2*i):
            H[m*l:(m+1)*l, :] = self.Y[:, m:m+j]
        H *= 1. / np.sqrt(j)
        return H

    def _T(self, block_rows):
        """Block toeplitz matrix from output correlations

        Arguments
        ---------
        block_rows : int
            Number of block rows

        Returns
        -------
        ndarray
            Block toeplitz matrix from output correlations
        """
        H = self._H(block_rows)
        k = H.shape[0] // 2
        Yf, Yp = H[k:], H[:k]
        return Yf.dot(Yp.T)

    def _weighting_matrices_balanced_realization(self, block_rows):
        """Weighting matrices W1 and W2 for the BR variant of SSI

        The balanced realization (BR) variant of the SSI algorithm
        uses the identity matrix as the weighing matrices.

        Arguments
        ---------
        block_rows : int
            Number of block rows

        Returns
        -------
        W1, W2 : 2darray, 2darray
            The balanced realization weighing matrices
        """
        W1 = np.eye(block_rows * self.Y.shape[0])
        W2 = W1.copy()
        return W1, W2

    def _weighting_matrices_canonical_variate_analysis(self, block_rows):
        """Weighting matrices W1 and W2 for the CVA variant of SSI

        The balanced realization (CVA) variant of the SSI algorithm
        uses the identity matrix as the weighing matrices.

        Arguments
        ---------
        block_rows : int
            Number of block rows

        Returns
        -------
        W1, W2 : 2darray, 2darray
            The canonical variate analysis weighting matrices
        """
        l, _ = self.Y.shape
        Tm = np.zeros((block_rows*l, block_rows*l))
        Tp = Tm.copy()
        for j in range(block_rows):
            for k in range(block_rows):
                i = np.abs(j + k)
                R = self._R(i)
                if j == k:
                    Tp[j*l:(j+1)*l, k*l:(k+1)*l] = R
                    Tm[j*l:(j+1)*l, k*l:(k+1)*l] = R
                elif j < k:
                    Tp[j*l:(j+1)*l, k*l:(k+1)*l] = R
                    Tm[j*l:(j+1)*l, k*l:(k+1)*l] = R.T
                else:
                    Tp[j*l:(j+1)*l, k*l:(k+1)*l] = R.T
                    Tm[j*l:(j+1)*l, k*l:(k+1)*l] = R
        Lm = np.linalg.cholesky(Tm)
        Lp = np.linalg.cholesky(Tp)
        I = np.eye(block_rows*l)
        W1 = np.linalg.solve(Lp, I)
        W2 = np.linalg.solve(Lm, I)
        return W1, W2

    def _weighting_matrices(self, block_rows):
        """Weighting matrices W1 and W2 for SSI.

        This method is called by the run method, subclass
        to create different variants of the CoVSSI method. See
        methods
            - _weighting_matrices_balanced_realization
            - _weighting_matrices_canonical_variate_analysis

        Arguments
        ---------
        block_rows : int
            Number of block rows

        Returns
        -------
        W1, W2 : 2darray, 2darray
            The weighting matrices
        """
        return self._weighting_matrices_balanced_realization(block_rows)

    @functools.lru_cache(maxsize=20, typed=False)
    def get_block_toeplitz_svd(self, block_rows):
        """Perform and return SVD of the block toeplitz matrix

        This method is cached, repeated calls with the same argument
        does not recompute the block toeplitz matrix.

        Arguments
        ---------
        block_rows : int
            Number of block rows

        Returns
        -------
        U, S, V : 2darray
            U and V are the unitary matrices and S is the
            singular values.
        """
        U, s, VT = np.linalg.svd(self._T(block_rows))
        return U, np.diag(s), VT.T

    def perform_block_toeplitz_svd(self, block_rows):
        """Perform SVD of the block toeplitz matrix and store in cache

        This method is cached, repeated calls with the same argument
        does not recompute the block toeplitz matrix.

        Arguments
        ---------
        block_rows : int
            Number of block rows

        See also
        --------
        get_block_toeplitz_svd

        """
        self.get_block_toeplitz_svd(block_rows)

    def find_system_matrices(self, block_rows, order):
        """Estimate system matrices A, C, G and R0 which generated the data

        Estimate the underlying model of the system that generated the data
        in the data array. The method returns the discrete state space matrix A,
        the output matrix C, the next state-output covariance matrix G, and the
        zero lag correlation matrix R0.

        Arguments
        ---------
        block_rows : int
            Number of block rows.
        order : int
            Order of the system.

        Returns
        -------
        A, C, G, R0 : ndarray
            State space matrix, output influence matrix, next state-output covariance
            matrix and zero lag correlation matrix.
        """
        l = self.Y.shape[0]
        logger.info("Performing SVD of block Toeplitz matrix")
        U, S, V = self.get_block_toeplitz_svd(block_rows)
        U1 = U[:, :order]
        S1 = S[:order, :order]
        V1 = V[:, :order]
        logger.info("Getting weighting matrices")
        W1, W2 = self._weighting_matrices(block_rows)
        logger.info("Establishing observability matrix")
        O = np.linalg.solve(W1, U1).dot(np.sqrt(S1))
        logger.info("Creating state space matrix and output matrix")
        Oup = O[:-l, :]
        Odn = O[l:, :]
        A = np.linalg.lstsq(Oup, Odn, rcond=None)[0]
        C = O[:l, :]
        logger.info("Establishing the controllability matrix")
        Gam = np.sqrt(S1).dot(V1.T).dot(np.linalg.solve(W2, np.eye(W2.shape[0])))
        logger.info("Creating the next state-output covariance matrix")
        G = Gam[:, -l:]
        R0 = self._R(0)
        return A, C, G, R0


class StateSpaceSystem:
    def __init__(self, A, C, G, R0, fs):
        """Discrete time state space system.

        This object provides interface to generate vibration modes
        from the discrete time state space system matrices.

        Arguments
        ---------
        A : 2darray
            Discrete time state space matrix.
        C : 2darray
            Output influence matrix.
        G : 2darray
            Next state-output covariance matrix.
        R0 : 2darray
            Zero lag correlation matrix.
        """
        self.A = A
        self.C = C
        self.G = G
        self.R0 = R0
        self.fs = fs
        self.modes = self._find_modes()

    def psd(self, f, N):
        """Find the power spectral density function of the system

        Calculate the power spectral density of the system with units
        (unit**2 / Hz).

        Arguments
        ---------
        f : 1darray
            Frequencies to estimate the psd at.
        N : int
            Number of sampled data points, used to scale the psd.

        Returns
        -------
        f, G : 1darray
            Frequencies and corrsponding psd.
        """
        w = 2 * np.pi * f
        dt = 1. / self.fs
        A, C, G, R0 = self.A, self.C, self.G, self.R0
        I = np.eye(self.order)
        S = np.zeros((w.size, *R0.shape), dtype=np.complex)
        for i, wi in enumerate(w):
            z = np.exp(1j*wi*dt)
            S[i] = C.dot(np.linalg.solve(z*I-A, G)) + R0 + G.T.dot(np.linalg.solve(I/z-A.T, C.T))
        Pyy = np.trace(S, axis1=1, axis2=2) * 2 * np.pi / np.sqrt(N)
        return Pyy.real

    @property
    def order(self):
        return self.A.shape[0]

    def _find_modes(self):
        "Returns a list of modes from the "
        lr, Q = np.linalg.eig(self.A)
        u = self.fs*np.log(lr)
        mask = u.imag > 0
        u = u[mask]
        Q = Q[:, mask]
        Phi = self.C.dot(Q)
        return [utils.Mode(ui, q) for ui, q in zip(u, Phi.T)]

    @property
    def natural_frequencies(self):
        return np.array([mode.f for mode in self.modes])

    @property
    def damped_natural_frequencies(self):
        return np.array([mode.fd for mode in self.modes])


class StabilizationDiagram:
    def __init__(self, state_space_systems):
        self.state_space_systems = sorted(state_space_systems, key=lambda x: x.order)

    def check_stability(self, a):
        pass
