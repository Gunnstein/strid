# -*- coding: utf-8 -*-
"""
Implementation of frequency domain identification methods:
    - pLSCF for identification of a right matrix fraction
      description of the system.
    - LSFD for identification of mode shapes from a FRF
      matrix and poles.
"""
import functools
import numpy as np

__all__ = ["PolyReferenceLSCF", ]


class PolyReferenceLSCF:
    """Poly reference Least Squares Complex Frequency-domain estimator

    Given a FRF matrix identify a right matrix fraction description
    (RMFD) of the system:

    .. math::

        H(\\omega_f) = N(\\omega_f)D^{-1}(\\omega_f)

    where :math:`H(\\omega)` is the FRF matrix,
    :math:`N(\\omega_f) = \\sum_{k=0}^n N_k \\Omega_f^k` is the numerator and
    :math:`D(\\omega_f) = \\sum_{k=0}^n D_k \\Omega_f^k` is the denominator
    matrix, :math:`\\Omega_f` is the basis function and :math:`n` is
    the model order.

    The denominator and numerator submatrices :math:`D_k` and
    :math:`N_k`, respectively are to be identified.

    Implementation is based on [Cauberghe2004].

    References
    ----------
    [Cauberghe2004] Cauberghe, B., 2004. Applied frequency-domain
        system identification in the field of experimental and
        operational modal analysis. PhD thesis. Vrije University,
        Brussel, Belgium.
    """
    def __init__(self, H, fs, sgn_exp_basis=1., constrain_N0=True):
        """Poly-reference least square complex frequency estimator

        Define a poly-reference least square complex frequency (pLSCF)
        frequency domain estimator. See [Cauberghe2004] for more information.

        Arguments
        ---------
        H : 3darray
            FRF matrix where the first and second axis refers to the
            outputs and inputs, respectively and the third axis refers to the 
            frequency.
        fs : float
            Sampling rate (Hz)
        sgn_exp_basis : {1, -1}, optional
            Sign of the exponent of the basis function, i.e. the basis
            is either :math:`z^{1}` or :math:`z^{-1}`. This parameter
            in combination with the `constrain_N0` parameter
            determines whether stochastic poles are modelled as stable
            or unstable poles and consequently affects the clarity of
            stabilization diagrams.
        constrain_N0 : bool, optional
            Determines if the 0th order denominator matrix is
            constrained to the unity matrix. If False, the maximum
            order matrix is constrained to the unity matrix. This
            parameter in combination with the `sgn_exp_basis`
            determines whether spurios poles are modelled as stable or
            unstable poles and consequently the clarity of
            stabilization diagrams.

        References
        ----------
        [Cauberghe2004] Cauberghe, B., 2004. Applied frequency-domain
            system identification in the field of experimental and
            operational modal analysis. PhD thesis. Vrije University,
            Brussel, Belgium.
        """
        self.H = H
        self.fs = fs
        self._max_order = 0
        self._SGN_EXP_BASIS = sgn_exp_basis / abs(sgn_exp_basis)
        self.constrain_N0 = constrain_N0

    @property
    def sgn_exp_basis(self):
        return self._SGN_EXP_BASIS

    @sgn_exp_basis.setter
    def sgn_exp_basis(self, v):
        raise AttributeError(
            "`sgn_exp_basis` cannot be changed after initialization.")

    @property
    def Ni(self):
        return self.H.shape[1]

    @property
    def No(self):
        return self.H.shape[0]

    @property
    def Nf(self):
        return self.H.shape[2]

    @property
    def f(self):
        return np.linspace(0., self.fs/2, self.Nf)

    @property
    def w(self):
        return 2*np.pi*self.f

    @property
    def _basis_function(self):
        SGN = self.sgn_exp_basis
        return np.exp(SGN*1j*self.w/self.fs)

    @functools.lru_cache(maxsize=128)
    def _X(self, order):
        Omega = self._basis_function
        return np.array([Omega**i for i in range(order+1)]).T

    @functools.lru_cache(maxsize=128)
    def _Y(self, order, o):
        X = self._X(order)
        Ho = self.H[o, :, :]
        return np.array([-np.kron(xo, Hoi) for xo, Hoi in zip(X, Ho.T)])

    @functools.lru_cache(maxsize=128)
    def _R(self, order):
        X = self._X(order)
        return (X.conj().T @ X).real

    @functools.lru_cache(maxsize=128)
    def _S(self, order, o):
        X = self._X(order)
        Y = self._Y(order, o)
        return (X.conj().T @ Y).real

    @functools.lru_cache(maxsize=128)
    def _T(self, order, o):
        Y = self._Y(order, o)
        return (Y.conj().T @ Y).real

    @functools.lru_cache(maxsize=5)
    def _M(self, order):
        M = np.zeros((self.Ni*(order+1), self.Ni*(order+1)))
        R = self._R(order)
        for o in range(self.No):
            S = self._S(order, o)
            T = self._T(order, o)
            M += T - S.T.conj() @ np.linalg.solve(R, S)
        return M

    def _alpha(self, order):
        """Solve for denominator matrices.

        Parameter redundancy means that a constraint must be imposed
        equation :math:`M \alpha = 0` for a non-trivial solution.
        Selection of the basis function and constraint coefficient
        determines whether deterministic and/or stochastic poles are
        modelled as stable and/or unstable poles(positive/negative
        damping).

        Arguments
        ---------
        order : int
            Order of the identified model, for the right fraction matrix
            description this is the order of the polynomial in numerator
            and denominator.

        Returns
        -------
        2darray
            Denominator matrices
        """
        n, Ni = order, self.Ni
        if order > self._max_order:
            self._max_order = order
        M = self._M(self._max_order)
        I = np.eye(Ni)
        if self.constrain_N0:
            alpha = np.r_[
                I,
                np.linalg.solve(-M[Ni:(n+1)*Ni, Ni:(n+1)*Ni], M[Ni:(n+1)*Ni, 0:Ni])]
        else:
            alpha = np.r_[
                np.linalg.solve(-M[0:n*Ni, 0:n*Ni], M[0:n*Ni, n*Ni:(n+1)*Ni]),
                I]
        return alpha

    def perform(self, order, max_order=50, return_numerator=True):
        """Perform system identification

        Arguments
        ---------
        order : int
            Order of the identified model, for the right fraction matrix
            description this is the order of the polynomial in numerator
            and denominator.
        max_order : int, optional
            Maximum order of any analysis for the object. Used to increase
            efficiency of the estimator since intermediate matrices for
            the maximum order must be established.
        return_numerator : bool, optional
            Whether or not to return the numerator matrices.

        Returns
        -------
        D or (N, D) : 3darrays
            Denominator and numerator arrays for the right matrix fraction
            description model. Where the first axis refers to the polynomial
            order, second axis the output and thrid axis to the input, i.e.
            N[i] and D[i] are the `i`th order matrices.
            D is only returned if `return_numerator` is true.
        """
        self._max_order = max_order
        alpha = self._alpha(order)
        D = alpha.reshape((-1, self.Ni, self.Ni))
        if return_numerator:
            R = self._R(order)
            beta = np.array([np.linalg.solve(-R, self._S(order, o) @ alpha)
                             for o in range(self.No)])
            N = np.moveaxis(beta, 1, 0)
            return N, D
        else:
            return D


def find_companion_matrix(D):
    """Find the (Frobenius) companion matrix

    Establish the companion matrix for the denominator matrices for
    determining the poles of a RMFD model.

    Arguments
    ---------
    D : 3darray
        Denominator matrix of RMFD model.

    Returns
    -------
    2darray
        Frobenius companion matrix.
    """
    p, Ni, _ = D.shape
    A = np.zeros((p*Ni, p*Ni))
    A[Ni:, :-Ni] = np.eye((p-1)*Ni)
    Dp = D[-1]
    for k, Dk in enumerate(D[:-1][::-1]):
        A[:Ni, k*Ni:(k+1)*Ni] = np.linalg.solve(-Dp, Dk)
    return A


def find_residues_lsfd(poles, H, fs):
    """Find residues from poles and FRF estimates

    Estimate the (in band) residue matrices from poles and FRF's by
    the Least Squares Frequency Domain Algorithm (LSFD).

    A residue matrix is the outer product of the mode vector
    and the modal participation factor. The mode vector can therefore
    be recovered by SVD decomposition of the residue matrix.

    Arguments
    ---------
    poles : 1darray
        Continous time poles (eigenvalues).
    H : 3darray
        FRF matrix where the first and second axis refers to the
            outputs and inputs, respectively and the third axis 
            refers to the frequency.
    fs : float
        Sampling rate

    Returns
    -------
    3darray
        Residue matrices where the first dimension refers
        to the poles, second dimension to outputs and
        third to inputs, i.e. if `R` is the returned matrix
        then `R[0]` is the residue matrix corresponding to
        pole `poles[0]`.
    """
    l, m, nf = H.shape
    p = np.r_[poles, poles.conj()]
    n = p.size
    A = np.zeros((l*nf, (n+2)*l), dtype=complex)
    w = 2*np.pi*np.linspace(0., fs/2, num=nf)
    I = np.eye(l)
    B = np.zeros((nf*l, m), dtype=complex)
    for i, wi in enumerate(w):
        A[i*l:(i+1)*l, -2*l:-1*l] = I / (1j*wi+1e-3)**1
        A[i*l:(i+1)*l, -l:] = I * (1j*wi)
        B[i*l:(i+1)*l, :] = H[:, :, i]
        for j, pj in enumerate(p):
            A[i*l:(i+1)*l, j*l:(j+1)*l] = I/(1j*wi-pj)
    X = np.linalg.lstsq(A, B, rcond=None)[0]
    return X[:l*n//2].reshape((n//2, l, m))
