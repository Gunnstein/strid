# -*- coding: utf-8 -*-
import numpy as np
import scipy.signal


__all__ = ["find_rayleigh_damping_coeffs", "get_frequency_vector",
           "modal_assurance_criterion", "w2f", "f2w", "norm2",
           "modal_scale_factor", "modal_phase_collinearity",
           "mean_phase", "mean_phase_deviation", "find_psd_matrix",
           "find_positive_psd_matrix", "find_frf_matrix",
           "accelerance2receptance", "receptance2accelerance", "Mode"]


def find_rayleigh_damping_coeffs(w, damping_ratios):
    """Rayleigh damping coefficients from target freqs and damping ratio

    Rayleigh damping is defined by the following relation

    .. math::

        C = aM + bK

    where :math:`C` is the damping matrix, :math:`M` is the mass
    matrix and :math:`K` is the stiffness matrix. The damping ratio
    :math:`\\xi` at angular frequency :math:`\\omega` is then given by
    the following equation:

    .. math::

       \\xi = \\frac{1}{2}(\\frac{a}{\\omega} + b\\omega)

    The damping coefficients is determined by specifying the desired
    damping ratio at two or more frequencies by the least square
    method.

    Arguments
    ---------
    w : 1darray
        Angular frequencies (rad/s) where the damping ratios are
        specified
    damping_ratios : 1darray
        The damping ratios (c / c_cr) at the specified frequencies.

    Returns
    -------
    a, b : float
        Rayleigh dampping coefficients
    """
    A = .5 * np.array([[1 / wn, wn]
                       for wn in w])
    return np.linalg.lstsq(A, damping_ratios, rcond=None)[0]


def w2f(w):
    "Convert angular frequency (rad/s) to frequency (Hz)"
    return w / (2*np.pi)


def f2w(f):
    "Convert frequency (Hz) to angular frequency (rad/s)"
    return f * (2*np.pi)


def modal_scale_factor(u, v):
    """Determine the scale factor between u and v

    Find the scale factor :math:`\\alpha` between :math:`u` and
    :math:`v` such that :math:`\\alpha u` has similar length and
    direction as :math:`v`.

    Argument
    --------
    u, v : 1darray
        Vectors to find the modal scale factor between

    Return
    ------
    complex
        The modal scale factor that makes the two vectors of
        similar length and phase.

    """
    u, v = np.asarray(u), np.asarray(v)
    return u.conj().dot(v) / np.linalg.norm(u)**2


def modal_phase_collinearity(u):
    """Modal phase collinearity of mode vector u

    Mode phase collinearity (MPC) quantifies the complexity of a
    mode vector. For classical normal modes, all dofs in a
    structure vibrate in phase with one another such that a maximum
    modal value is reached at the same instant for all dofs. MPC
    ranges from 0 to 1, where MPC=1 indicates that all dofs are in
    phase, and zero indicates out of phase dofs.

    Arguments
    ---------
    u : 1darray[complex]
        Mode shape vector

    Returns
    -------
    float
        Modal phase collinearity of the mode vector

    References
    ----------
    [Pappa1992] Pappa RS, Elliott KB, Schenk A (1992)
        A consistent-mode indicator for the eigensystem realization
        algorithm. NASA Report TM-107607
    """
    u = np.asarray(u)
    S = np.cov(u.real, u.imag)
    l = np.linalg.eigvals(S)
    return (l[0]-l[1])**2/(l[0]+l[1])**2


def mean_phase(u):
    """Mean phase of mode vector u

    Mean phase (MP) is the angle of a linear line fitted through the
    mode shape in the complex plane.

    Arguments
    ---------
    u : 1darray[complex]
        Mode shape vector

    Returns
    -------
    float
        Mean phase of the mode vector

    References
    ----------
    [Reynders2012] Reynders, E., Houbrechts, J., De Roeck, G., 2012.
        Fully automated (operational) modal analysis. Mechanical
        Systems and Signal Processing 29, 228–250.
        https://doi.org/10.1016/j.ymssp.2012.01.007
    """
    u = np.asarray(u)
    U, s, VT = np.linalg.svd(np.c_[u.real, u.imag])
    V = VT.T
    return np.arctan(-V[0, 1] / V[1, 1])


def mean_phase_deviation(u):
    """Mean phase deviation of mode vector u

    Mean phase deviation (MPD) is the deviation in phase from the mean
    phase and is a measure of mode shape complexity.

    Arguments
    ---------
    u : 1darray[complex]
        Mode shape vector

    Returns
    -------
    float
        Mean phase deviation of the mode vector

    References
    ----------
    [Reynders2012] Reynders, E., Houbrechts, J., De Roeck, G., 2012.
        Fully automated (operational) modal analysis. Mechanical
        Systems and Signal Processing 29, 228–250.
        https://doi.org/10.1016/j.ymssp.2012.01.007
    """
    u = np.asarray(u)
    U, s, VT = np.linalg.svd(np.c_[u.real, u.imag])
    V = VT.T
    w = np.abs(u)
    num = u.real*V[1, 1] - u.imag*V[0, 1]
    den = np.sqrt(V[0, 1]**2+V[1, 1]**2)*np.abs(u)
    return np.sum(w*np.arccos(np.abs(num/den))) / np.sum(w)


def norm2(v):
    """Return the Euler norm of v.

    Return the Euler norm :math:`||v||_2`.

    Arguments
    ---------
    v : 1darray
        Vector to find the norm of.

    Returns
    -------
    float : Euler norm of v.
    """
    return np.linalg.norm(v, 2)


def get_frequency_vector(fs, n):
    return np.linspace(0., fs/2, n)


def modal_assurance_criterion(u, v):
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
    u = np.asarray(u)
    v = np.asarray(v)
    return np.abs(u.conj().T@v)**2 / (u.conj().T@u*v.conj().T@v).real


def find_psd_matrix(x, y, **kwargs):
    """Calculate PSD matrix for x and y.

    Arguments
    ---------
    x, y : 2darray
       Measurement matrix where each row corresponds to the entire time
       series of a measurement channel.
    kwargs : dict, optional
       All keyword arguments are passed to the scipy.signal.csd,
       see docstring of scipy.signal.csd.

    Returns
    -------
    3darray
       PSD matrix where the first and second axis refers to the
       degree of freedom of `x` and `y`, respectively and the third axis
       refers to the frequency.

    See Also
    --------
    find_positive_psd_matrix :
        Positive lag psd matrix for frequency based operational modal
        analysis.
    """
    Pxy = np.array(
        [[scipy.signal.csd(xi, yj, **kwargs)[1]
          for yj in y] for xi in x])
    return Pxy


def find_positive_psd_matrix(x, y, nfft=2**9, window="rectangular"):
    """Calculate positive lag PSD matrix

    Calculate the positive lag PSD matrix as per [Cauberghe2004] (pg.
    47, fig. 3.8). Useful in combined deterministic-stochastic or
    stochastic frequency domain modal extraction methods such as LSCF
    and pLSCF.

    Arguments
    ---------
    x, y : 2darray
       Measurement matrix where each row corresponds to the entire time
       series of a measurement channel.
    nfft : int, optional
        Length of the block/segment used in estimation of PSD. Note that
        the number of samples from `x` and `y`, is half of `nfft`, i.e.
        `nperseg=nfft//2` and the segment is then zero padded.
    window : {'rectangular', 'exponential'}, optional
        Which window to apply to remove the negative lag coefficients.

    Returns
    -------
    3darray
       PSD matrix where the first and second axis refers to the
       degree of freedom of `x` and `y`, respectively and the third axis
       refers to the frequency.

    Raises
    ------
    ValueError
        Raised if `window` parameter is not `rectangular` or `exponential`.

    See Also
    --------
    find_psd_matrix :
        Conventional psd matrix.
    """
    Pxy = find_psd_matrix(x, y,
                          nperseg=nfft//2,
                          nfft=nfft,
                          noverlap=0,
                          window="boxcar",)
    Rxy = np.fft.irfft(Pxy)
    if window == "rectangular":
        win = scipy.signal.boxcar(Rxy.shape[2])
        win[Rxy.shape[2]//2:] *= 0.
    elif window == "exponential":
        tau = - Rxy.shape[2] / np.log(.01)
        win = scipy.signal.exponential(
            Rxy.shape[2], center=0, tau=tau, sym=False)
    else:
        raise ValueError(
            "`window` must be either `rectangular` or exponential`"
        )
    Rxy *= win
    return np.fft.rfft(Rxy)


def find_frf_matrix(u, y, estimator="H1", **kwargs):
    """Estimate the FRF matrix from input and output.

    Estimate the FRF matrix from input `u` and output `y`.

    Arguments
    ---------
    u, y : 2darray
        Input and output where each row contains the data points
        for a particular channel.

    estimator : str
        FRF estimator can be either `H1` or `H2`. The `H2`
        estimator requires an equal number of input and outputs
        to exist.

    kwargs :
       All keyword arguments are passed to the scipy.signal.csd,
       see docstring.

    Returns
    -------
    3darray
       FRF matrix where the first and second axis refers to the
       degree of freedom of `y` and `u`, respectively and the third axis
       refers to the frequency.

    Raises
    ------
    ValueError
        If the `H2` estimator is selected, but the number of inputs does not
        match the number of outputs.
    ValueError
        If estimator is not `H1` or `H2`.
    """
    if estimator.upper() == "H1":
        Sr = find_psd_matrix(u, u, **kwargs)
        Sl = find_psd_matrix(y, u, **kwargs)
    elif estimator.upper() == "H2":
        if u.shape[0] != y.shape[0]:
            raise ValueError(
                "# of inputs must match # of outputs for `H2` estimator."
            )
        Sr = find_psd_matrix(u, y, **kwargs)
        Sl = find_psd_matrix(y, y, **kwargs)
    else:
        raise ValueError(
            "FRF estimator must be either `H1` or `H2`."
        )
    H = np.moveaxis(np.array([
        np.linalg.solve(Sr[:, :, i].T, Sl[:, :, i].T).T
        for i in range(Sr.shape[2])]), 0, 2)
    return H


def accelerance2receptance(H, fs):
    """Convert accelerance to receptance.

    Convert the FRF matrix of accelerations, i.e. accelerance to a
    FRF matrix of displacements, i.e. receptance.

    .. math::

        H_u(\\omega) = \frac{H_\\ddot{u}(\\omega)}{\\omega^{2}}

    The operation is done in place and no value is returned.

    Arguments
    ---------
    H : 3darray
        Accelerance, first axis refers to the output,
        second axis to the input and third axis to the
        frequency.
    fs : float
        Sampling rate.
    """
    w = 2*np.pi*np.linspace(0., fs/2, H.shape[2])
    for i, wi in enumerate(w):
        if wi == 0.:
            continue
        H[:, :, i] /= -wi**2


def receptance2accelerance(H, fs):
    """Convert accelerance to receptance.

    Convert the FRF matrix of displacement, i.e. receptance to a FRF
    matrix of accelerations, i.e. accelerance.

    .. math::

        H_\\ddot{u}(\\omega) = H_u(\\omega) \\omega^{2}

    The operation is done in place and no value is returned.

    Arguments
    ---------
    H : 3darray
        Receptance, first axis refers to the output,
        second axis to the input and third axis to the
        frequency.
    fs : float
        Sampling rate.
    """
    w = 2*np.pi*np.linspace(0., fs/2, H.shape[2])
    for i, wi in enumerate(w):
        H[:, :, i] *= -wi**2


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
        K = np.zeros((n, n), float)
        for i in range(n):
            K[i, i] = 2 * k
            if i > 0:
                K[i-1, i] = -k
            if i < n-1:
                K[i+1, i] = -k
        K[-1, -1] = k
        return K

    def get_natural_frequency(self, r):
        """Returns the analytical natural frequency of mode `r`

        Arguments
        ---------
        r : int
            Mode to return the frequency for.

        Returns
        -------
        float
            Natural frequency of mode `r` in rad/s
        """
        k, m, n = self.k, self.m, self.n
        return 2 * np.sqrt(k / m) * np.sin(np.pi / 2 * (2*r-1) / (2*n+1))

    def get_mode_shape(self, r):
        """Returns the analytical mode shape of mode `r`

        Arguments
        ---------
        r : int
            Mode to return the mode shape for.

        Returns
        -------
        1darray
            Mode shape of mode `r`, the mode shape is normalized
            to have unit length.
        """
        x = np.array([np.sin(i*np.pi*(2*r-1)/(2*self.n+1))
                      for i in range(1, self.n+1)])
        return x / norm2(x)

    def set_rayleigh_damping_matrix(self, freqs, xis):
        """Set the damping matrix to the Rayleigh damping matrix

        Rayleigh damping is a classical damping matrix which defines
        the modal damping ratio x(w) by the following equation

            xi(w) = 1/2 * (a/w + b*w)

        where w is the frequency in rad/s and a and b are the Rayleigh
        damping coefficients. Specify the damping ratio for atleast two
        frequencies, if more than two damping ratios are specified, the
        Rayleigh damping coefficients are determined by least squre fitting.

        Arguments
        ---------
        freqs : 1darray
            Frequencies (rad/s) where the damping ratios are specified
        damping_ratios : 1darray
            The damping ratios (c / c_cr) at the specified frequencies.
        """
        a, b = find_rayleigh_damping_coeffs(freqs, xis)
        self._rayleigh_coeffs = (a, b)
        self.C = a*self.M + b*self.K

    def get_rayleigh_damping_ratio(self, r):
        """Returns the Rayleigh damping ratio of mode `r`

        It is assumed that Rayleigh damping is set, see
        set_rayleigh_damping_matrix method.

        Arguments
        ---------
        r : int
            Mode to return the damping ratio for.

        Returns
        -------
        float
            Damping ratio of mode `r`
        """

        a, b = self._rayleigh_coeffs
        w = self.get_natural_frequency(r)
        return .5*(a/w+b*w)

    @property
    def state_matrix(self):
        "CT State space  matrix (B)"
        M, C, K = self.M, self.C, self.K
        Z = np.zeros_like(M)
        I = np.eye(M.shape[0])
        A11 = -np.linalg.solve(M, C)
        A12 = -np.linalg.solve(M, K)
        A = np.r_[np.c_[A11, A12],
                  np.c_[I, Z]]
        return A

    @property
    def input_influence_matrix(self):
        "CT State space input influence matrix (B)"
        return np.r_[np.linalg.solve(self.M, np.eye(self.n)),
                     np.zeros((self.n, self.n))]

    def get_state_space_matrices(self):
        "Continous time state space matrices A, B, C, D"
        A = self.state_matrix
        B = self.input_influence_matrix
        n = self.n
        O = np.zeros((n, n))
        I = np.eye(n)
        C = np.r_[A[:n, :],
                  np.c_[I, O],
                  np.c_[O, I]]
        D = np.r_[np.linalg.solve(self.M, I),
                  O,
                  O]
        return A, B, C, D

    def simulate(self, t, F=None, d0=None, v0=None):
        """Obtain system response to load and initial conditions.

        Simulate the system response at time points `t` due to loads
        `F` and with initial displacements `d0` and velocities `v0`.

        Arguments
        ---------
        t : 1darray
            Time points to evaluate the system response.
        F : 2darray, optional
            Load matrix where each column corresponds to time points in
            `t` and each row is the load applied to a system dof. Fij is
            then the load applied to dof `i` at time `j`. Zeros is assumed
            if None.
        d0, v0 : 1darray, optional
            Initial displacment and velocity vector. Zeros is assumed
            if None.

        Returns
        -------
        A, V, D : 2darray
            Acceleration, velocity and displacement vector for the system.

        """
        n = self.n
        d0 = np.zeros(n) if d0 is None else d0
        v0 = np.zeros(n) if v0 is None else v0
        x0 = np.r_[v0, d0]

        sys = scipy.signal.StateSpace(*self.get_state_space_matrices())
        U = np.zeros((t.size, n)) if F is None else F.T

        _, y, _ = scipy.signal.lsim(sys, U, t, X0=x0)
        y = y.T
        A = y[:n, :]
        V = y[n:2*n, :]
        D = y[2*n:, :]
        return A, V, D


def rmfd2ss(N, D):
    """Convert RMFD model to state space model

    Convert a right matrix fraction description (RMFD)
    matrices N and D to state space matrices A, B, C, D.
    See [Reynders2012].

    Arguments
    ---------
    N, D : 3darrays
        Numerator and denominator matrices for RMFD model.

    Returns
    -------
    A, B, C, D : 2darrays
        State space matrices.

    References
    ----------
    [Reynders2012] Reynders, E., 2012. System Identification Methods
        for (Operational) Modal Analysis: Review and Comparison. Arch
        Computat Methods Eng 19, 51–124.
        https://doi.org/10.1007/s11831-012-9069-x
    """
    n, l, m = N.shape
    A = np.zeros((n*m, n*m))
    A[m:, :-m] = np.eye((n-1)*m)
    C = np.zeros((l, n*m))
    Np = N[-1]
    Dp = D[-1]
    for k, (Dk, Nk) in enumerate(zip(D[:-1][::-1], N[:-1][::-1])):
        DpDk = np.linalg.solve(Dp, Dk)
        A[:m, k*m:(k+1)*m] = -DpDk
        C[:, k*m:(k+1)*m] = Nk - Np@DpDk
    B = np.zeros((n*m, m))
    B[:m, :] = np.linalg.solve(Dp, np.eye(m))
    D = np.linalg.solve(Dp.T, Np.T).T
    return A, B, C, D


class Mode(object):
    def __init__(self, eigenvalue, eigenvector):
        """Mode converts eigenvalue/vector to vibration mode characteristics

        A mode defines a single degree of freedom dynamic system
        with frequency (f and w) and damping (xi) characteristics and mode
        shape vector (v).

        Arguments
        ---------
        eigenvalue : complex
            Eigenvalue (in continous time) of the mode.
        eigenvector : 1darray[float or complex]
            Eigenvector or modal vector.
        """
        self.eigenvalue = eigenvalue
        self.eigenvector = eigenvector

    @property
    def v(self):
        return self.eigenvector / np.linalg.norm(self.eigenvector, 2)

    @property
    def w(self):
        return self.eigenvalue.imag / np.sqrt(1-self.xi**2)

    @property
    def wd(self):
        return self.eigenvalue.imag

    @property
    def xi(self):
        u = self.eigenvalue
        return -u.real / np.abs(u)

    @property
    def f(self):
        return self.w / (2.0*np.pi)

    @property
    def fd(self):
        return self.wd / (2.*np.pi)

    @property
    def mean_phase_colinearity(self):
        return modal_phase_collinearity(self.v)

    @property
    def mpc(self):
        return self.mean_phase_colinearity

    @property
    def mean_phase(self):
        return mean_phase(self.v)

    @property
    def mp(self):
        return self.mean_phase

    @property
    def mean_phase_deviation(self):
        return mean_phase_deviation(self.v)

    @property
    def mpd(self):
        return self.mean_phase_deviation

    @classmethod
    def find_modes_from_ss(cls, A, C, fs):
        """Return modes from the (discrete) system matrices A and C

        This method finds all modes from the discrete state space system
        matrices A and C.

        Arguments
        ---------
        A : 2darray
            Discrete time state space matrix
        C : 2darray
            Output influence matrix
        fs : float
            Sampling rate

        Returns
        -------
        list
            List of modes (Mode objects)
        """
        lr, Q = np.linalg.eig(A)
        u = fs*np.log(lr)
        Phi = C.dot(Q)
        return [cls(ui, q) for ui, q in zip(u, Phi.T)]

    @classmethod
    def find_modes_from_rmfd(cls, N, D, fs):
        """Return modes from a the right matrix fraction description (RMFD).

        This method returns modes from the RMFD numerator and denominator
        matrices N and D.

        Arguments
        ---------
        N, D : 3darrays
            Numerator and denominator matrices for RMFD.
        fs : float
            Sampling rate

        Returns
        -------
        list
            List of modes (Mode objects)
        """
        A, B, C, D = rmfd2ss(N, D)
        return cls.find_modes_from_ss(A, C, fs)
