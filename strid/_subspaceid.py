# -*- coding: utf-8 -*-
"""
Implementation of stochastic and combined deterministic-stochastic subspace
identification methods
"""
import abc
import functools
import numpy as np
import scipy.linalg
from .utils import find_psd_matrix, get_frequency_vector


__all__ = ["CombinedDeterministicStochasticSID",
           "DataDrivenStochasticSID",
           "CovarianceDrivenStochasticSID",
           "AbstractReferenceBasedStochasticSID"]


def create_block_hankel_matrix(data, block_rows, ncols=None, ix_ref=None):
    """Block hankel matrix from data array

    Arguments
    ---------
    data : 2darray
        Data array where each row contains data from one
        sensor and each column corresponds to a specific
        time.
    block_rows : int
        Number of block rows
    ncols : int, optional
        Number of columns in block hankel matrix. If None,
        all data in the data matrix is used.
    ix_ref : list, optional
        Indices to the reference outputs in y. If `None`, all outputs
        are considered to be references.

    Returns
    -------
    2darray
        Block hankel matrix from data array
    """
    l, s = data.shape
    ix_ref = ix_ref or [*range(l)]
    r = len(ix_ref)
    i = block_rows
    j = ncols or s - 2*i + 1
    y = data
    yref = y[ix_ref]
    H = np.zeros(((r+l)*i, j))
    for m in range(2*i):
        if m < i:
            H[m*r:(m+1)*r, :] = yref[:, m:m+j]
        else:
            H[r*i+(m-i)*l:r*i+(m+1-i)*l, :] = y[:, m:m+j]
    return 1./np.sqrt(j)*H


class AbstractReferenceBasedStochasticSID(abc.ABC):
    @abc.abstractmethod
    def __init__(self, y, fs, ix_references=None):
        """Subspace identificator

        Arguments
        ---------
        y : 2darray
            Output data matrix (l x s) from `l` outputs with `s` samples.
        fs : float
            Sampling rate
        ix_references : list, optional
            Indices to the reference outputs in y. If `None`, all outputs
            are considered to be references.
        """
        self.y = y
        self.fs = fs
        self.ix_references = ix_references or [*range(self.l)]

    @abc.abstractmethod
    def perform(self, *args, **kwargs):
        pass

    def psdy(self, return_trace=False, **kw):
        """Compute power spectral density matrix of outputs

        Compute the power spectral density matrix of the outputs
        with Welch's method.

        Arguments
        ---------
        return_trace : bool, optional
            Return the entire psd matrix or only the trace of the psd matrix.
        kw : dict
            See keywords to scipy.signal.csd and scipy.signal.welch.

        Returns
        -------
        f : 1darray
            Frequency vector of the psd matrix
        Pyy : 3darray or 1darray
            Output PSD matrix where the first dimension refers to the
            frequency of the psd estimator, see get_frequency_vector,
            and the second and third dimensions refers to the degree
            of freedom of the input and output as given in y. If
            return_trace, the trace of the psd matrix is returned
            instead of the entire psd matrix.
        """
        psd = find_psd_matrix(self.y, self.y, fs=self.fs, **kw)
        f = get_frequency_vector(self.fs, psd.shape[2])
        if return_trace:
            out = np.trace(psd)
        else:
            out = psd
        return f, out

    @property
    def yref(self):
        return self.y[self.ix_references]

    @property
    def l(self):
        return self.y.shape[0]

    @property
    def r(self):
        return len(self.ix_references)

    @property
    def s(self):
        return self.y.shape[1]

    def j(self, i):
        """Number of columns in block hankel matrices

        Returns the number of columns in the block hankel
        matrices given that all the data is used in the
        identification problem

        Arguments
        ---------
        i : int
            Number of block rows

        Returns
        -------
        int
            Number of columns in the block hankel
            matrices.
        """
        return self.s-2*i+1

    @functools.lru_cache(maxsize=20, typed=False)
    def _Y(self, i):
        """Output block hankel matrix

        Arguments
        ---------
        i : int
            Number of block rows

        Returns
        -------
        2darray
            Output block hankel matrix
        """
        return create_block_hankel_matrix(
            self.y, i, ix_ref=self.ix_references)


class CovarianceDrivenStochasticSID(AbstractReferenceBasedStochasticSID):
    """Stochastic subspace identificator

    Given measurements of output :math:`y` identify the system
    matrices :math:`A, C`, and the covariance matrices :math:`Q, R, S`
    of the process noise :math:`w` and measurement noise :math:`v` for
    the system.

    .. math::

        x_{k+1} = Ax_k + w_k
        y_k = Cx_k + v_k


    Implementation is based on [Overschee1996] and [Peeters1999].

    References
    ----------
    [Overschee1996] Van Overschee, P., De Moor, B., 1996.
        Subspace Identification for Linear Systems.
        Springer US, Boston, MA. doi: 10.1007/978-1-4613-0465-4

    [Peeters1999] Peeters, B., De Roeck, G., 1999.
        Reference based stochastic subspace identification
        for output only modal analysis.
        Mechanical Systems and Signal Processing 13, 855–878.
        doi: 10.1006/mssp.1999.1249
    """
    def __init__(self, y, fs, ix_references=None):
        """Reference-based covariance-driven stochastic subspace identicator.

        Define a reference-based covariance driven stochastic subspace
        identificator (SSI-cov/ref). See [Overschee1996] and
        [Peeters1999] for more information.

        Arguments
        ---------
        y : 2darray
            Output data matrix (l x s) from `l` outputs with `s` samples.
        fs : float
            Sampling rate (Hz)
        ix_references : list, optional
            Indices to the reference outputs in y. If `None`, all outputs
            are considered to be references.

        References
        ----------
        [Overschee1996] Van Overschee, P., De Moor, B., 1996.
            Subspace Identification for Linear Systems.
            Springer US, Boston, MA. doi: 10.1007/978-1-4613-0465-4

        [Peeters1999] Peeters, B., De Roeck, G., 1999.
            Reference based stochastic subspace identification
            for output only modal analysis.
            Mechanical Systems and Signal Processing 13, 855–878.
            doi: 10.1006/mssp.1999.1249
        """
        self.y = y
        self.fs = fs
        self.ix_references = ix_references or [*range(self.l)]

    @functools.lru_cache(maxsize=20, typed=False)
    def _R(self, lag):
        """Correlation matrix of data array

        Correlation matrix between data array with `lag` last samples
        removed and data array with first `lag` samples removed.

        Arguments
        ---------
        lag : int
            Time lag / shift

        Returns
        -------
        2darray
            Correlation matrix
        """
        s = self.s
        i = np.abs(lag)
        return self.y[:, :s-i].dot(self.yref[:, i:].T) / (s-i)

    @functools.lru_cache(maxsize=20, typed=False)
    def _T(self, i):
        """Block toeplitz matrix from output correlations

        Arguments
        ---------
        i : int
            Number of block rows

        Returns
        -------
        2darray
            Block toeplitz matrix from output correlations
        """
        Y = self._Y(i)
        Yp = Y[:self.r*i]
        Yf = Y[self.r*i:]
        return Yf @ Yp.T

    @functools.lru_cache(maxsize=20, typed=False)
    def _svd_block_toeplitz(self, i):
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
        U, s, VH = np.linalg.svd(self._T(i))
        return U, s, VH

    def perform(self, order, block_rows):
        """Perform system identification

        Arguments
        ---------
        order : int
            Order of the identified model
        block_rows : int
            Number of block rows

        Returns
        -------
        A, C, G, R0 : 2darrays
            System, output, next state-output covariance
            and zero lag correlation matrix.

        Raises
        ------
        ValueError
            The ratio between the order and number of block rows must
            be less or equal than the number of references to ensure a
            consistent equation system, i.e. the system has valid
            dimensions.
        """
        i, l, n, r = block_rows, self.l, order, self.r
        if n/i > r:
            raise ValueError(
                "Following condition violated: order / block_rows <= r")
        U, s, VH = self._svd_block_toeplitz(i)
        U1 = U[:, :n]
        V1H = VH[:n]
        sqrt_S1 = np.diag(np.sqrt(s[:n]))
        inv_sqrt_S1 = np.diag(1/np.sqrt(s[:n]))

        Oi = U1 @ sqrt_S1
        C = Oi[:l]

        Cref = sqrt_S1 @ V1H
        G = Cref[:, -r:]

        T1 = self._T(i)
        T2 = np.zeros_like(T1)
        T2[:-l, :] = T1[l:, :]
        T2[-l:, r:] = T1[-l:, :-r]
        T2[-l:, :r] = self._R(2*i)
        A = inv_sqrt_S1 @ U1.T @ T2 @ V1H.T @ inv_sqrt_S1

        R0 = self.y @ self.y.T / self.s
        return A, C, G, R0


class DataDrivenStochasticSID(AbstractReferenceBasedStochasticSID):
    """Stochastic subspace identificator

    Given measurements of output :math:`y` identify the system
    matrices :math:`A`, :math:`C`, and the covariance matrices
    :math:`Q, R, S` of the process noise :math:`w` and measurement
    noise :math:`v` for the system

    .. math::

        x_{k+1} = Ax_k + w_k
        y_k = Cx_k + v_k
    """
    def __init__(self, y, fs, ix_references=None):
        """Reference-based data-driven stochastic subspace identicator.

        Define a reference-based data driven stochastic subspace identificator
        (SSI-data/ref). See [Overschee1996] and [Peeters1999] for more
        information.

        Arguments
        ---------
        y : 2darray
            Output data matrix (l x s) from `l` outputs with `s` samples.
        fs : float
            Sampling rate (Hz)
        ix_references : list, optional
            Indices to the reference outputs in y. If `None`, all outputs
            are considered to be references.

        References
        ----------
        [Overschee1996] Van Overschee, P., De Moor, B., 1996.
            Subspace Identification for Linear Systems.
            Springer US, Boston, MA. doi: 10.1007/978-1-4613-0465-4

        [Peeters1999] Peeters, B., De Roeck, G., 1999.
            Reference based stochastic subspace identification
            for output only modal analysis.
            Mechanical Systems and Signal Processing 13, 855–878.
            doi: 10.1006/mssp.1999.1249
        """
        self.y = y
        self.fs = fs
        self.ix_references = ix_references or [*range(self.l)]

    @functools.lru_cache(maxsize=20, typed=False)
    def _LQ_decomposition_block_hankel(self, i):
        """LQ decomposition of data block-hankel matrix

        The projections can be performed more numerically efficient, accurate
        and stable by utilizing the LQ decomposition of the data
        matrix to determine the projections and other operations.

        Arguments
        ---------
        i : int
            Number of block rows

        Returns
        -------
        2darray
            Lower triangular matrix from LQ decomposition of
            data block hankel matrix.
        """
        QT, LT = scipy.linalg.qr(self._Y(i).T, mode='economic')
        Q = QT.T
        L = LT.T
        return L, Q

    @functools.lru_cache(maxsize=20, typed=False)
    def _svd_weighted_projection(self, i):
        """Perform SVD of the projection matrix

        Arguments
        ---------
        block_rows : int
            Number of block rows

        Returns
        -------
        U : 2darray
            U is a unitary matrix with the left singular vectors.
        s : 1darray
            Singular values of projection matrix.
        """
        L, Q = self._LQ_decomposition_block_hankel(i)
        Lpi = L[self.r*i:, :self.r*i]
        U, s, _ = np.linalg.svd(Lpi)
        return U, s


    def perform(self, order, block_rows):
        """Perform system identification

        Arguments
        ---------
        order : int
            Order of the identified model, note `order / block_rows <= r`
            where `r` is the number of reference outputs.
        block_rows : int
            Number of block rows, note `order / block_rows <= r`
            where `r` is the number of reference outputs.

        Returns
        -------
        A, C, G, R0 : 2darrays
            System, output, next state-output covariance
            and zero lag correlation matrix.

        Raises
        ------
        ValueError
            The ratio between the order and number of block rows must
            be less or equal than the number of references to ensure a
            consistent equation system, i.e. the system has valid
            dimensions.
        """
        i, j, l, n, r = block_rows, self.j(block_rows), self.l, order, self.r
        if n/i > r:
            raise ValueError(
                "Following condition violated: order / block_rows <= r")
        L, Q = self._LQ_decomposition_block_hankel(i)
        U, s = self._svd_weighted_projection(i)
        Epinv = np.diag(1/np.sqrt(s[:n])) @ U[:, :n].T
        X = Epinv @ L[r*i:, :r*i] # Actually X*Q1

        Oim1 = U[:-l, :n] @ np.diag(np.sqrt(s[:n]))
        ix_row_4 = [*range((r+l)*i-l*(i-1), (r+l)*i)]
        ix_row_2_and_3 = [*range(r*i, r*i+l)]
        XY = np.zeros((n+l, r*i+l))
        XY[n:] = L[ix_row_2_and_3, :r*i+l]
        XY[:n, :r*(i+1)] = np.linalg.lstsq(
            Oim1, L[ix_row_4, :r*(i+1)], rcond=None)[0]

        AC = np.linalg.lstsq(X.T, XY[:, :r*i].T, rcond=None)[0].T
        A = AC[:n]
        C = AC[n:]

        # Compute next state and zero lag covariance
        algorithm = 1
        if algorithm == 1:
            R0 =(1/j*L[r*i:r*i+l, :r*i+l] @ L[r*i:r*i+l, :r*i+l].T)
            G = (1/j*Epinv @ L[r*i:, :r*i] @ L[:r*i, :r*i].T)[:, -r:]
        elif algorithm == 3:
            # This is the algorithm presented in
            # [Peeters1999] and algorithm 3 in [Overschee1996],
            # however, the residuals are zero in the implementation
            # and therefore R0 and G are as well.
            # Using algorithm 1 from [Overschee1996] until error is
            # corrected.
            r = XY@Q[:r*i+l] - AC@X@Q[:r*i]
            QRS = np.cov(r)
            Q, R, S = QRS[:n, :n], QRS[n:, n:], QRS[:n, n:]
            Sig = scipy.linalg.solve_discrete_lyapunov(A, Q)
            R0 = C @ Sig @ C.T + R
            G = A @ Sig @ C.T + S

        return A, C, G, R0

class CombinedDeterministicStochasticSID(AbstractReferenceBasedStochasticSID):
    """Combined deterministic stochastic subspace identificator

    Given measurments of output :math:`y` and input :math:`u` identify
    the system matrices :math:`A, B, C, D` and the covariance matrices
    :math:`Q, R, S` of the process noise `w` and measurement noise `v`
    for the system

    .. math::

        x_{k+1} = Ax_k + Bu_k + w_k
        y_k = Cx_k + Du_k + v_k

    """
    def __init__(self, u, y, fs, ix_references=None):
        """Reference-based combined deterministic-stochastic subspace identicator.

        Define a reference-based combined deterministic-stochastic subspace
        identificator (CSI/ref).

        See [Overschee1996] and [Reynders2008] for more information.

        Arguments
        ---------
        u : 2darray
            Input data matrix (m x s) from `m` input with `s` samples.
        y : 2darray
            Output data matrix (l x s) from `l` outputs with `s` samples.
        fs : float
            Sampling rate (Hz)
        ix_references : list, optional
            Indices to the reference outputs in y. If `None`, all outputs
            are considered to be references.

        References
        ----------
        [Overschee1996] Van Overschee, P., De Moor, B., 1996.
            Subspace Identification for Linear Systems.
            Springer US, Boston, MA. doi: 10.1007/978-1-4613-0465-4

        [Reynders2008] Reynders, E., Roeck, G.D., 2008.
            Reference-based combined deterministic–stochastic subspace
            identification for experimental and operational modal analysis.
            Mechanical Systems and Signal Processing 22, 617–637.
            odi: 10.1016/j.ymssp.2007.09.004

        """
        self.u = u
        self.y = y
        self.fs = fs
        self.ix_references = ix_references or [*range(self.l)]

    @property
    def m(self):
        return self.u.shape[0]

    @functools.lru_cache(maxsize=20, typed=False)
    def _U(self, i):
        """Input block hankel matrix

        Arguments
        ---------
        i : int
            Number of block rows

        Returns
        -------
        2darray
            Input block hankel matrix
        """
        return create_block_hankel_matrix(self.u, i)

    def psdu(self, return_trace=False, **kw):
        """Compute power spectral density matrix of inputs

        Compute the power spectral density matrix of the inputs
        with Welch's method.

        Arguments
        ---------
        return_trace : bool, optional
            Return the entire psd matrix or only the trace of the psd matrix.
        kw : dict
            See keywords to scipy.signal.csd and scipy.signal.welch.

        Returns
        -------
        f : 1darray
            Frequency vector of the psd matrix
        Puu : 3darray or 1darray
            Input PSD matrix where the first dimension refers to the
            frequency of the psd estimator, see get_frequency_vector,
            and the second and third dimensions refers to the degree
            of freedom of the input and output as given in y. If
            return_trace, the trace of the psd matrix is returned
            instead of the entire psd matrix.
        """
        psd = find_psd_matrix(self.u, self.u, fs=self.fs, **kw)
        f = get_frequency_vector(self.fs, psd.shape[2])
        if return_trace:
            out = np.trace(psd)
        else:
            out = psd
        return f, out

    @functools.lru_cache(maxsize=20, typed=False)
    def _LQ_decomposition_block_hankel(self, i):
        """LQ decomposition of data block-hankel matrix

        The projections can be performed more numerically efficient, accurate
        and stable by utilizing the LQ decomposition of the input-output
        matrix to determine the projections and other operations.

        Arguments
        ---------
        i : int
            Number of block rows

        Returns
        -------
        2darray
            Lower triangular matrix from LQ decomposition of
            data block hankel matrix.
        """
        W = np.vstack((self._U(i), self._Y(i)))
        R = scipy.linalg.qr(W.T, mode='r')[0]
        K = min(R.shape)
        L = R[:K, :].T
        return L

    @functools.lru_cache(maxsize=20, typed=False)
    def _svd_weighted_projection(self, i):
        """Perform SVD of the projection matrix

        Arguments
        ---------
        block_rows : int
            Number of block rows

        Returns
        -------
        U : 2darray
            U is a unitary matrix with the left singular vectors.
        s : 1darray
            Singular values of projection matrix.
        """
        U, s, _ = np.linalg.svd(self._weighted_projection(i))
        return U, s

    def _weighted_projection(self, i):
        """Return weighted oblique projection of input-output

        Arguments
        ---------
        i : int
            Number of block rows

        Returns
        -------
        2darray
            Weighted oblique projection of input-output
        """
        l, m, r = self.l, self.m, self.r
        L = self._LQ_decomposition_block_hankel(i)
        Lint = np.linalg.solve(L[:i*(2*m+r), :i*(2*m+r)].T, L[i*(2*m+r):, :i*(2*m+r)].T).T
        LUp = Lint[:, :i*m]
        LYpref = Lint[:, -i*r:]
        L23_13 = L[i*m:2*i*m, :2*i*m]
        Pi = np.eye(2*m*i) - L23_13.T @ np.linalg.solve(L23_13 @ L23_13.T, L23_13)
        W1_Oi_W2 = np.hstack(
            ((LUp @ L[:i*m, :2*i*m] + LYpref @ L[2*i*m:i*(2*m+r), :2*i*m]) @ Pi,
              LYpref @ L[2*i*m:i*(2*m+r), 2*i*m:i*(2*m+r)]))
        return W1_Oi_W2

    def perform(self, order, block_rows, estimate_B_and_D=False, estimate_covariances=False):
        """Perform system identification

        Arguments
        ---------
        order : int
            Order of the identified model, note `order / block_rows <= r`
            where `r` is the number of reference outputs.
        block_rows : int
            Number of block rows, note `order / block_rows <= r`
            where `r` is the number of reference outputs.
        estimate_B_and_D : bool, optional
            Estimate and return the input matrix `B` and the direct feedthrough
            matrix `D`.
        estimate_covariances : bool, optional
            Estimate and return the covariance matrix `Q` of the
            process noise (stochastic load), covariance matrix `R` of
            the measurement noise and the covariance matrix `S`
            between process and measurement noise.

        Returns
        -------
        A : 2darray
            System matrix of state space model.
        B : 2darray, optional
            Input matrix of state space model, only returned if `estimate_B_and_D`
            is True.
        C : 2darray
            Output matrix of state space model.
        D : 2darray, optional
            Direct feedthrough matrix of state space model, only returned if
            `estimate_B_and_D` is True.
        Q : 2darray, optional
            Covariance matrix of the process noise, only returned if
            `estimate_covariances` is True.
        R : 2darray, optional
            Covariance matrix of the measurement noise, only returned if
            `estimate_covariances` is True.
        S : 2darray, optional
            Covariance matrix between the process and measurement noise,
            only returned if `estimate_covariances` is True.

        Raises
        ------
        ValueError
            The ratio between the order and number of block rows must
            be less or equal than the number of references to ensure a
            consistent equation system, i.e. the system has valid
            dimensions.
        """
        i, l, m, n, r = block_rows, self.l, self.m, order, self.r
        if n/i > r:
            raise ValueError(
                "Following condition violated: order / block_rows <= r")
        L = self._LQ_decomposition_block_hankel(i)
        U, s = self._svd_weighted_projection(i)
        Gam_i = U[:, :n] @ np.diag(np.sqrt(s[:n]))
        pinv_Gam_i = np.diag(1./np.sqrt(s[:n])) @ U[:, :n].T
        Gam_im1 = Gam_i[:-l]
        Tl = np.vstack(
            (np.linalg.lstsq(Gam_im1, L[-(i-1)*l:, :-(i-1)*l], rcond=None)[0],
             L[-i*l:-(i-1)*l, :-(i-1)*l]))
        Tr = np.vstack(
            (pinv_Gam_i @ L[-i*l:, :-(i-1)*l],
             L[i*m:2*i*m, :-(i-1)*l]))

        S = np.linalg.lstsq(Tr.T, Tl.T, rcond=None)[0].T

        # Find A and C
        A = S[:n, :n]
        C = S[n:, :n]

        if estimate_B_and_D:
            # Approach using the LQ result
            # ("robust algorithm", [VanOverschee1996])
            L_zero = np.hstack((L[-i*l:, :i*(2*m+r)], np.zeros((i*l, l))))
            P = Tl - S[:, :n] @ pinv_Gam_i @ L_zero

            L1 = A @ pinv_Gam_i
            L2 = C @ pinv_Gam_i
            M = np.linalg.pinv(Gam_im1)
            Il = np.zeros((l, l))
            N1 = np.vstack(
                (np.hstack([L1[:, :l]] + [M[:, (k-1)*l:k*l] - L1[:, k*l:(k+1)*l] for k in range(1, i)]),
                 np.hstack([Il-L2[:, :l]] + [-L2[:, k*l:(k+1)*l] for k in range(1, i)])))
            Nk = np.zeros_like(N1)
            Nr = np.vstack((
                np.hstack((np.eye(l), np.zeros((l, Gam_im1.shape[1])))),
                np.hstack((np.zeros((Gam_im1.shape[0], l)), Gam_im1))
            ))

            UN = np.zeros(((i*(2*m+r)+l)*(n+l), m*(n+l)))
            for k in range(i):
                Ufk = L[m*(k+i):m*(k+i+1), :-(i-1)*l]
                Nk[:, :(i-k)*l] = N1[:, k*l:]
                Nk[:, (i-k)*l:] = 0.
                UN += np.kron(Ufk.T, Nk @ Nr)

            DB = np.linalg.lstsq(UN, P.T.reshape(-1), rcond=None)[0].reshape((m, -1)).T
            D = DB[:l]
            B = DB[l:]
        else:
            B = None
            D = None

        # Find covariance matrices
        if estimate_covariances:
            TTT = Tl - Tl @ np.linalg.lstsq(Tr, Tr, rcond=None)[0]
            QSR = TTT @ TTT.T
            Q = QSR[:n, :n]
            S = QSR[n:, :n]
            R = QSR[n:, n:]
        else:
            Q = None
            S = None
            R = None

        ret = tuple(mat for mat in [A, B, C, D, Q, R, S] if mat is not None)
        return ret
