# -*- coding: utf-8 -*-
import numpy as np
import scipy.signal

__all__ = ["ShearFrame", "find_rayleigh_damping_coeffs",
           "get_frequency_vector", "find_modal_assurance_criterion",
           "find_psd_matrix",]


def find_rayleigh_damping_coeffs(freqs, damping_ratios):
    """Rayleigh damping coefficients from target freqs and damping ratio

    Rayleigh damping is defined by the following relation

        C = a*M + b*K

    where C is the damping matrix, M is the mass matrix and K is the
    stiffness matrix. The damping ratio `xi` at frequency `f` is then
    given by the following equation:

       xi = 1/2 * (a/(2*pi*f) + b * (2*pi*f))

    The damping coefficients can be determined by specifying the desired
    damping ratio at two or more frequencies. The least square method is
    used to determine the damping ratios.

    Arguments
    ---------
    freqs : 1darray
        Frequencies (rad/s) where the damping ratios are specified
    damping_ratios : 1darray
        The damping ratios (c / c_cr) at the specified frequencies.
    """
    A = .5 * np.array([[1 / wn, wn]
                       for wn in freqs])
    return np.linalg.lstsq(A, damping_ratios, rcond=None)[0]


def get_frequency_vector(fs, n):
    return np.linspace(0., fs/2, n)


def find_modal_assurance_criterion(u, v):
    """Calculate the modal assurance criterion (MAC)

    MAC is the square of the linear correlation between two mode shapes
    and is a measure of the similarity of two different modes which varies
    between 0 and 1. MAC=1 means a perfect correlation between two vectors
    exists, while MAC=0 means no linear correlation.

    Arguments
    ---------
    u, v : 1darray
        Mode shapes to check the MAC for.

    Returns
    -------
    float
        MAC between two vectors u and v.
    """
    H = lambda x: np.conjugate(x).T
    return np.abs(H(u).dot(v))**2 / (H(u).dot(u).real * H(v).dot(v).real)


def find_psd_matrix(Y, **kwargs):
    """Calculate the PSD matrix from the measured data A

    Arguments
    ---------
    Y : ndarray
       Measurement matrix where each row corresponds to the entire time
       series of a measurement channel.
    kwargs :
       All keyword arguments are passed to the scipy.signal.csd,
       see docstring.

    Returns
    -------
    ndarray
       PSD (n x m x m) matrix where the first dimension refers to the
       frequency of the psd estimator, see get_frequency_vector, and
       the second and third dimensions refers to the degree of freedom
       of the input and output as given in Y.
    """
    Pyy = np.array(
        [[scipy.signal.csd(yi, yj, **kwargs)[1]
          for yj in Y] for yi in Y]).T
    return Pyy


class ShearFrame(object):
    def __init__(self, n, m, k):
        """Create a shear frame object

        Define the shear frame with the mass `m` of each floor,
        stiffness `k` of each column, and the number of storeys
        `n`, see figure below.

                    m          DOF
                ========= --->  n
                |       |
              k |       | k
                |   m   |
                ========= --->  n-1
                |       |
              k |       | k
                |   m   |
                ========= --->  n-2
                |       |
                :       :
                :       :
                |   m   |
                ========= --->  3
                |       |
              k |       | k
                |   m   |
                ========= --->  2
                |       |
              k |       | k
                |   m   |
                ========= --->  1
                |       |
              k |       | k
                |       |
               +++     +++


        The natural frequencies and the modeshapes of this system
        can be determined analytically. The `r`th natural frequency
        of a shear frame is defined by

            f_r = 2 * sqrt(k / m) * sin(p * (2r-1) / 2) / (2n+1)

        and the `i`th element of the `r`th mode shape is defined by

            phi =  sin(i *pi*(2r-1)/(2n+1))

        Arguments
        ---------
        n : int
            Number of storeys and also the number of DOF of the dynamic
            system.
        m : float
            Mass of each floor
        k : float
            Stiffness of each column in each storey.
        """
        self.n = n
        self.m = m
        self.k = k
        self.M = np.eye(n) * m
        self.K = self._get_K()
        self.C = np.zeros_like(self.M)

    def _get_K(self):
        k, n = self.k, self.n
        K = np.zeros((n, n), np.float)
        for i in range(n):
            K[i, i] = 2 * k
            if i > 0:
                K[i-1, i] = -k
            if i < n-1:
                K[i+1, i] = -k
        K[-1, -1] = k
        return K

    def find_natural_frequency(self, r):
        k, m, n = self.k, self.m, self.n
        return 2 * np.sqrt(k / m) * np.sin(np.pi / 2 * (2*r-1) / (2*n+1))

    def find_mode_shape(self, r):
        x = np.array([np.sin(i*np.pi*(2*r-1)/(2*self.n+1))
                      for i in range(1, self.n+1)])
        return x / np.abs(x).max()

    def set_rayleigh_damping_matrix(self, freqs, xis):
        """Set the damping matrix to the Rayleigh damping matrix

        Arguments
        ---------
        freqs : 1darray
            Frequencies (rad/s) where the damping ratios are specified
        damping_ratios : 1darray
            The damping ratios (c / c_cr) at the specified frequencies.
        """
        a, b = find_rayleigh_damping_coeffs(freqs, xis)
        self.C = a*self.M + b*self.K

    def find_state_matrix(self):
        M, C, K = self.M, self.C, self.K
        Z = np.zeros_like(M)
        I = np.eye(M.shape[0])
        A11 = -np.linalg.solve(M, C)
        A12 = -np.linalg.solve(M, K)
        B11 = np.linalg.solve(M, I)
        A = np.r_[np.c_[A11, A12],
                  np.c_[I, Z]]
        B = np.r_[np.c_[B11, Z],
                  np.c_[Z, I]]
        return A, B

    def get_transfer_func(self, w, eigenvalues=None):
        if eigenvalues is None:
            A, _ = self.find_state_matrix()
            eigenvalues, _ = np.linalg.eig(A)
        D = lambda s: 1 / np.prod(np.complex(imag=s)-eigenvalues)
        G = np.array([np.abs(D(wi)) for wi in w])
        return G
