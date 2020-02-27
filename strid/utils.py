# -*- coding: utf-8 -*-
import numpy as np
import scipy.signal
import scipy.integrate

__all__ = ["find_rayleigh_damping_coeffs",
           "get_frequency_vector", "find_modal_assurance_criterion",
           "find_psd_matrix", "w2f", "f2w", "norm2", ]


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


def w2f(w):
    "Convert angular frequency (rad/s) to frequency (Hz)"
    return w / (2*np.pi)

def f2w(f):
    "Convert frequency (Hz) to angular frequency (rad/s)"
    return w * (2*np.pi)


def norm2(v):
    "Return the euler norm (||v||_2) of vector `v`"
    return np.linalg.norm(v, 2)


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

            w_r = 2 * sqrt(k / m) * sin(p * (2r-1) / 2) / (2n+1)

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

    def get_natural_frequency(self, r):
        "Returns the analytical natural frequency of mode `r`"
        k, m, n = self.k, self.m, self.n
        return 2 * np.sqrt(k / m) * np.sin(np.pi / 2 * (2*r-1) / (2*n+1))

    def get_mode_shape(self, r):
        "Returns the analytical mode shape of mode `r`"
        x = np.array([np.sin(i*np.pi*(2*r-1)/(2*self.n+1))
                      for i in range(1, self.n+1)])
        return x / norm2(x)

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

    @property
    def state_matrix(self):
        M, C, K = self.M, self.C, self.K
        A11 = -np.linalg.solve(M, C)
        A12 = -np.linalg.solve(M, K)
        Z = np.zeros_like(M)
        I = np.eye(M.shape[0])
        A = np.r_[np.c_[A11, A12],
                  np.c_[I, Z]]
        return A
    
    def solve(self, t, F=None, d0=None, v0=None, method="RK45"):
        """Obtain system response to load and initial conditions. 
        
        Solve the system at time points `t` due to loads `F` and with
        initial displacements `d0` and velocities `v0`.
        
        Arguments
        ---------
        t : 1darray
            Time points to evaluate the system response.
        F : Optional[2darray]
            Load matrix where each column corresponds to time points in
            `t` and each row is the load applied to a system dof. Fij is
            then the load applied to dof `i` at time `j`. Zeros is assumed
            if None.
        d0, v0 : Optional[1darray]
            Initial displacment and velocity vector. Zeros is assumed
            if None.
        method : str
            Defines the solver used to get the system response, see
            scipy.integrate.solve_ivp.
            
        Returns
        -------
        A, V, D : 2darray
            Acceleration, velocity and displacement vector for the system.
        """
        A = self.state_matrix
        
        d0 = np.zeros(A.shape[0]) if d0 is None else d0
        v0 = np.zeros(A.shape[0]) if v0 is None else v0
        x0 = np.r_[v0, d0]
        
        if F is None:
            U = np.zeros((self.n, t.size))
        else:
            U = np.linalg.solve(self.M, F)
            
        def Bu(ti):
            return np.array([0.]*self.n + [np.interp(ti, t, u) for u in U])
                 
        def dydt(t, x):
            return A.dot(x) + Bu(t)
        
        result = scipy.integrate.solve_ivp(dydt, [t[0], t[-1]], x0, t_eval=t, method=method)
        X = result.y
        V = X[:self.n, :]
        D = X[self.n:, :]
        Acc = A.dot(X)[:self.n, :] + U
        return Acc, V, D

