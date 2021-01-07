# -*- coding:utf-8 -*-
"""
Implementation of deterministic, stochastic and combined subspace identification methods

The implementation and terminology is based on the following resources

References
----------

[Overschee1996] Van Overschee, P., De Moor, B.,
    Subspace Identification for Linear Systems.
    Springer US, Boston, MA. doi: 10.1007/978-1-4613-0465-4

[Rainieri2014] Rainieri, C., Fabbrocino, G., 2014. Operational Modal Analysis of
    Civil Engineering Structures. Springer. doi: 10.1007/978-1-4939-0767-0
"""
import abc
import functools
import numpy as np
import scipy.linalg


__all__ = ["CombinedDeterministicStochasticSID", ]


def create_block_hankel_matrix(data, block_rows, ncols=None):
    """Block hankel matrix from data array

    Arguments
    ---------
    data : 2darray
        Data array where each row contains data from one
        sensor and each column corresponds to a specific
        time.
    block_rows : int
        Number of block rows
    ncols : Optional[int]
        Number of columns in block hankel matrix. If None,
        all data in the data matrix is used.

    Returns
    -------
    ndarray
        Block hankel matrix from data array
    """
    l, s = data.shape
    i = block_rows
    j = ncols or s - 2*i + 1
    H = np.zeros((l*2*i, j))
    for m in range(2*i):
        H[m*l:(m+1)*l, :] = data[:, m:m+j]
    return 1/np.sqrt(j)*H


class BaseSID(abc.ABC):
    @abc.abstractmethod
    def __init__(self, u, y):
        """Subspace identificator

        Arguments
        ---------
        u : 2darray
            Input data matrix (m x s) from `m` input with `s` samples.
        y : 2darray
            Output data matrix (l x s) from `l` outputs with `s` samples.
        """
        self.y = y
        self.u = u

    @abc.abstractmethod
    def perform(self, *args, **kwargs):
        pass

    @property
    def l(self):
        return self.y.shape[0]

    @property
    def s(self):
        return self.y.shape[1]

    @property
    def m(self):
        return self.u.shape[0]

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
        return create_block_hankel_matrix(self.y, i)

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

    def _Yp(self, i):
        """Output block hankel matrix of past

        Arguments
        ---------
        i : int
            Number of block rows

        Returns
        -------
        2darray
            Output block hankel matrix of past
        """
        return self._Y(i)[:i*self.l]

    def _Yf(self, i):
        """Output block hankel matrix of future

        Arguments
        ---------
        i : int
            Number of block rows

        Returns
        -------
        2darray
            Output block hankel matrix of future
        """
        return self._Y(i)[i*self.l:]

    def _Up(self, i):
        """Input block hankel matrix of past

        Arguments
        ---------
        i : int
            Number of block rows

        Returns
        -------
        2darray
            Input block hankel matrix of past
        """
        return self._U(i)[:i*self.m]

    def _Uf(self, i):
        """Input block hankel matrix of future

        Arguments
        ---------
        i : int
            Number of block rows

        Returns
        -------
        2darray
            Input block hankel matrix of future
        """
        return self._U(i)[i*self.m:]

    def _Ypp(self, i):
        """Output block hankel matrix of past (plus)

        Arguments
        ---------
        i : int
            Number of block rows

        Returns
        -------
        2darray
            Output block hankel matrix of past (plus)
        """
        return self._Y(i)[:(i+1)*self.l]

    def _Yfm(self, i):
        """Output block hankel matrix of future (minus)

        Arguments
        ---------
        i : int
            Number of block rows

        Returns
        -------
        2darray
            Output block hankel matrix of future (minus)
        """
        return self._Y(i)[(i+1)*self.l:]

    def _Upp(self, i):
        """Input block hankel matrix of past (plus)

        Arguments
        ---------
        i : int
            Number of block rows

        Returns
        -------
        2darray
            Input block hankel matrix of past (plus)
        """
        return self._U(i)[:(i+1)*self.m]

    def _Ufm(self, i):
        """Input block hankel matrix of future (minus)

        Arguments
        ---------
        i : int
            Number of block rows

        Returns
        -------
        2darray
            Input block hankel matrix of future (minus)
        """
        return self._U(i)[(i+1)*self.m:]

    def _Wp(self, i):
        """Input-output block hankel matrix of past

        Arguments
        ---------
        i : int
            Number of block rows

        Returns
        -------
        2darray
            Input-output block hankel matrix of past
        """
        return np.vstack((self._Up(i), self._Yp(i)))

    def _Wpp(self, i):
        """Input-output block hankel matrix of past (plus)

        Arguments
        ---------
        i : int
            Number of block rows

        Returns
        -------
        2darray
            Input-output block hankel matrix of past (plus)
        """
        return np.vstack((self._Upp(i), self._Ypp(i)))

    def _W(self, i):
        """Input-output block hankel matrix

        Arguments
        ---------
        i : int
            Number of block rows

        Returns
        -------
        2darray
            Input-output block hankel matrix
        """
        return np.vstack((self._U(i), self._Y(i)))


class CombinedDeterministicStochasticSID(BaseSID):
    """Combined deterministic stochastic subspace identificator

    Given measurments of output `y` and input `u` identify the
    system matrices A, B, C, D and the covariance matrices Q, R, S
    of the process noise `w` and measurement noise `v` for the system

    .. math::

        x_{k+1} = Ax_k + Bu_k + w_k
        y_k = Cx_k + Du_k + v_k


    Implementation is based on the numerically robust algorithm for
    combined deterministic-stochastic subspace identification presented
    in [Overschee1996].

    References
    ----------
    [Overschee1996] Van Overschee, P., De Moor, B., 1996.
        Subspace Identification for Linear Systems.
        Springer US, Boston, MA. doi: 10.1007/978-1-4613-0465-4
    """
    def __init__(self, u, y):
        """Combined deterministic-stochastic subspace identificator

        Arguments
        ---------
        u : 2darray
            Input data matrix (m x s) from `m` input with `s` samples.
        y : 2darray
            Output data matrix (l x s) from `l` outputs with `s` samples.
        """
        self.y = y
        self.u = u

    @functools.lru_cache(maxsize=20, typed=False)
    def _R_from_RQ_decomposition(self, i):
        """Return R from RQ decomposition of input-output BH matrix

        The projections can be performed more numerically efficient, accurate
        and stable by utilizing the RQ decomposition of the input-output
        matrix to determine the projections and other operations.

        Arguments
        ---------
        i : int
            Number of block rows

        Returns
        -------
        2darray
            Lower triangular matrix from RQ decomposition of input-output
            block hankel matrix.

        Note
        ----
        The RQ decomposition as refered to in [Overschee1996] is
        elsewhere known as the LQ decomposition where R=L is a lower
        triangular matrix.

        References
        ----------
        [Overschee1996] Van Overschee, P., De Moor, B., 1996.
            Subspace Identification for Linear Systems.
            Springer US, Boston, MA. doi: 10.1007/978-1-4613-0465-4

        """
        R = scipy.linalg.qr(self._W(i).T, mode='r')[0]
        K = min(R.shape)
        L = R[:K, :].T
        return L

    def _ix_block_rows_R(self, i):
        j, l, m = self.j(i), self.l, self.m
        return np.cumsum([0, m*i, m, m*(i-1), l*i, l, l*(i-1)])

    def _weighted_oblique_projection(self, i):
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
        j, l, m, s = self.j(i), self.l, self.m, self.s
        R = self._R_from_RQ_decomposition(i)
        ix = self._ix_block_rows_R(i)
        L = R[ix[4]:ix[6], ix[0]:ix[4]].dot(
            scipy.linalg.pinv(R[ix[0]:ix[4], ix[0]:ix[4]]))
        LUp = L[:, :m*i]
        LYp = L[:, -l*i:]
        RPi = R[ix[1]:ix[3], ix[0]:ix[3]]
        Pi = (np.eye(2*m*i)
              - RPi.T.dot(np.linalg.lstsq(RPi.dot(RPi.T), RPi, rcond=None)[0]))
        O1 = ((LUp.dot(R[ix[0]:ix[1], ix[0]:ix[3]])
               + LYp.dot(R[ix[3]:ix[4], ix[0]:ix[3]]))).dot(Pi)
        O2 = LYp.dot(R[ix[3]:ix[4], ix[3]:ix[4]])
        return np.hstack((O1, O2))

    @functools.lru_cache(maxsize=20, typed=False)
    def _svd_weighted_oblique_projection(self, i):
        """Return the SVD of the weighted oblique projection

        Arguments
        ---------
        i : int
            Number of block rows

        Returns
        -------
        U, s, VH : 2darray, 1darray, 2darray
            SVD of weighted oblique projection of input-output
        """
        O = self._weighted_oblique_projection(i)
        U, s, VH = np.linalg.svd(O, full_matrices=False)
        return U, s, VH

    def perform(self, order, block_rows, estimate_B_and_D=False, estimate_covariances=False):
        """Perform system identification

        Arguments
        ---------
        order : int
            Order of the identified model
        block_rows : int
            Number of block rows
        estimate_B_and_D : Optional[bool]
            Estimate and return the input matrix `B` and the direct feedthrough
            matrix `D`.
        estimate_covariances : Optional[bool]
            Estimate and return the covariance matrix `Q` of the
            process noise (stochastic load), covariance matrix `R` of
            the measurement noise and the cross covariance matrix `S`
            between process and measurement noise.

        Returns
        -------
        A, C : 2darrays
            System and output matrices of the state space model.
            If `estimate_B_and_D=False` and `estimate_covariances=False`.
        A, B, C, D : 2darrays
            System, input, output and direct feedthrough matrices
            of the state space model. If `estimate_B_and_D=True` and
            `estimate_covariances=False`.
        A, B, Q, R, S : 2darrays
            System and output matrices of the state space model and
            the covariances of the process and measurement noise.
            If `estimate_B_and_D=False` and `estimate_covariances=True`.
        A, B, C, D, Q, R, S: 2darrays
            System, input, output and direct feedthrough matrices
            of the state space model  and the covariances of the
            process and measurement noise. If `estimate_B_and_D=True` and
            `estimate_covariances=True`.
        """
        n, i, j, l, m, s = order, block_rows, self.j(block_rows), self.l, self.m, self.s
        R = self._R_from_RQ_decomposition(i)
        ix = self._ix_block_rows_R(i)
        U, s, VH = self._svd_weighted_oblique_projection(i)
        Gi = U[:, :n].dot(np.diag(np.sqrt(s[:n])))
        Gim1 = Gi[:-l, :]
        Tl = np.vstack(
            (np.linalg.lstsq(Gim1, R[ix[5]:ix[6], ix[0]:ix[5]], rcond=None)[0],
             R[ix[4]:ix[5], ix[0]:ix[5]]))
        Tr = np.vstack(
            (np.linalg.lstsq(Gi, R[ix[4]:ix[6], ix[0]:ix[5]], rcond=None)[0],
             R[ix[1]:ix[3], ix[0]:ix[5]]))
        S = Tl.dot(scipy.linalg.pinv(Tr))

        # Find A and C
        A = S[:n, :n]
        C = S[n:,:n]

        # Find B and D
        if estimate_B_and_D:
            P = Tl - S[:, :n].dot(
                np.linalg.lstsq(Gi, R[ix[4]:ix[6], ix[0]:ix[5]], rcond=None)[0])
            Q = R[ix[1]:ix[3], ix[0]:ix[5]]

            L = S[:, :n].dot(scipy.linalg.pinv(Gi))
            L1 = lambda k: L[:n, (k-1)*l:k*l]  # in R^{n\times l}
            L2 = lambda k: L[n:, (k-1)*l:k*l]  # in R^{l\times l}

            Mmat = scipy.linalg.pinv(Gim1)
            M = lambda k: Mmat[:, (k-1)*l:k*l] # in R^{n\times l}

            N11 = np.hstack(
                [-L1(1)] + [M(k-1)-L1(k) for k in range(2, i+1)]) # in R^{n \times li}
            N21 = np.hstack(
                [np.eye(l)-L2(1)] + [-L2(k) for k in range(2, i+1)]) # in R^{l \times li}

            QN = np.zeros((P.size, m*(l+n)), dtype=np.float)
            Nl = np.zeros((n+l, l*i))
            Nr = np.vstack((
                np.hstack((np.eye(l), np.zeros((l, Gim1.shape[1])))),
                np.hstack((np.zeros((Gim1.shape[0], l)), Gim1))
            ))

            for k in range(i):
                Qk = Q[k*m:(k+1)*m, :]
                Nl *= 0.
                Nl[:n, :l*(i-k)] = N11[:, :l*(i-k)]
                Nl[n:, :l*(i-k)] = N21[:, :l*(i-k)]
                Nk = Nl.dot(Nr)
                QNk = np.kron(Qk.T, Nk)
                QN += QNk
            vecDB = np.linalg.lstsq(QN, P.T.reshape(-1), rcond=None)[0]
            DB = vecDB.reshape((m, -1)).T
            D = DB[n:, :]
            B = DB[:n, :]
        else:
            B = None
            D = None

        # Find covariance matrices
        if estimate_covariances:
            E = (Tl-S.dot(Tr))
            QSR = E.dot(E.T)
            Q = QSR[:n, :n]
            S = QSR[n:, :n]
            R = QSR[n:, n:]
        else:
            Q = None
            S = None
            R = None
        ret = (mat for mat in [A, B, C, D, Q, R, S] if mat is not None)
        return ret
