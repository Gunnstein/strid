# -*- coding: utf-8 -*-
import numpy as np


__all__ = ["ShearFrame", ]


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

    @staticmethod
    def find_rayleigh_coeffs(frequencies, dampingratios):
        A = np.array([[1 / (2*wn), .5 * wn] for wn in frequencies])
        return np.linalg.lstsq(A, dampingratios)[0]

    def set_rayleigh_damping_matrix(self, freqs, xis):
        a, b = self.find_rayleigh_coeffs(freqs, xis)
        self.C = a*self.M + b*self.K

    def find_state_matrix(self):
        M, C, K = self.M, self.C, self.K
        A11 = -np.linalg.solve(M, C)
        A12 = -np.linalg.solve(M, K)
        B11 = np.linalg.solve(M, np.eye(M.shape[0]))
        Z = np.zeros_like(M)
        I = np.eye(M.shape[0])
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
