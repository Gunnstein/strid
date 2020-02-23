# -*- coding: utf-8 -*-
import numpy as np


__all__ = ["generalized_alpha_method", "generalized_alpha_method_bf"]


class GeneralizedSystem(object):
    """Generalized system for the generalized alpha algorithm

    This helper class provides an interface to the generalized system
    matrices used in solving a structural system with the generalized
    alpha algorithm.
    """
    def __init__(self, M, C, K, alpha_f=0.5, alpha_m=0.5, beta=0.5,
                 gamma=0.25):
        self.M = M
        self.C = C
        self.K = K
        self.alpha_f = alpha_f
        self.alpha_m = alpha_m
        self.beta = beta
        self.gamma = gamma

    def Kdyn(self, dt):
        return (
            (1.0-self.alpha_m)*self.M
            + (1.0-self.alpha_f)*self.gamma*dt*self.C
            + (1.0-self.alpha_f)*self.beta*dt**2*self.K)

    def Mf(self, dt):
        return (
            self.alpha_m*self.M
            + (1.0-self.alpha_f)*(1-self.gamma)*dt*self.C
            + (1.0-self.alpha_f)*(0.5-self.beta)*dt**2*self.K
        )

    def Cf(self, dt):
        return self.C + (1-self.alpha_f)*dt*self.K

    def Kf(self, dt):
        return self.K


def generalized_alpha_method_bf(M, C, K, F, dt, d0, v0, alpha_f=.5,
                                 alpha_m=.5, beta=0.5, gamma=0.25):
    """Generalized alpha method in basic form.

    The generalized alpha algorithm implements a family of time
    integration methods commonly used in solving structural dynamics
    problems, i.e. systems on the form:

        Ma(t) + Cv(t) + Kd(t) = f(t)

    where v and a are the first and second derivative of d wrt time.
    Included is the well known HHT-alpha (alpha_m=0),
    WBZ-alpha (alpha_f=0)  and Newmark method (alpha_f=alpha_m=0).

    Stability
    ---------
    The algorithm is unconditionally stable for linear systems provided
    that

        alpha_m <= alpha_f <= 1/2

    and

        beta > 1/4 + 1/2*(alpha_f-alpha_m)

    Accuracy
    --------
    The algorithm is second-order accurate provided that

        gamma = 1/2 - alpha_m + alpha_f

    Arguments
    ---------
    M, C, K, F : ndarray
        Mass, Damping and Stiffness matrices defining the structural
        system and the load matrix where each column is the load vector
        f(t) at time t.
    dt : float
        Time increment.
    d0, v0 : 1darray
        Initial displacement and velocity vectors, respectively.
    alpha_f, alpha_m, beta, gamma : Optional[float]
        Algorithmic parameters which determine the characteristics of
        the algorithm. The default values corresponds to constant
        acceleration integrator with no algorithmic damping.

    Returns
    -------
    A, V, D : ndarray
        Acceleration, velocity and displacement matrices for the
        solved system.

    Note
    ----
    The system is solved with a general purpose solver
    (np.linalg.solve) at each step. The performance of this
    implementation can be greatly improved by changing the solver to a
    specialized solver when appropriate, e.g. a sparse solver for a
    sparse system.

    References
    ----------
        J. Chung, G. M. Hulbert. A time integration Algorithm for
        Structural Dynamics With Improved Numerical Dissipation:
        The Generalized-alpha method.
        Journal of Applied Mechanics (1993) vol. 60, pg 371-375.

    """
    generalized_system = GeneralizedSystem(M, C, K, alpha_f=alpha_f,
                                           alpha_m=alpha_m, beta=beta,
                                           gamma=gamma)
    Kdyn = generalized_system.Kdyn(dt)
    Mf = generalized_system.Mf(dt)
    Cf = generalized_system.Cf(dt)
    Kf = generalized_system.Kf(dt)

    D = np.zeros_like(F)
    V = np.zeros_like(D)
    A = np.zeros_like(D)
    D[:, 0] = d0
    V[:, 0] = v0
    A[:, 0] = np.linalg.solve(M, F[:, 0]-C.dot(v0)-K.dot(d0))

    af, am, b, g = alpha_f, alpha_m, beta, gamma
    for n in range(F.shape[1]-1):
        f = (1-af)*F[:, n+1] + af*F[:, n]
        fdyn = (f - Mf.dot(A[:, n]) - Cf.dot(V[:, n]) - Kf.dot(D[:, n]))

        A[:, n+1] = np.linalg.solve(Kdyn, fdyn)
        D[:, n+1] = (D[:, n]
                     + dt*V[:, n]
                     + dt**2*((.5-b)*A[:, n] + b*A[:, n+1]))
        V[:, n+1] = V[:, n] + dt*((1-g)*A[:, n] + g*A[:, n+1])
    return A, V, D


def generalized_alpha_method(M, C, K, F, dt, d0, v0, rho=1.0):
    """Solve equation of motion with the generalized-alpha method

    The generalized alpha method is an implicit integration algorithm
    controlled by a single parameter, `rho`. `rho` governs the amount
    of numerical damping introduced by the method. In the present
    formulation, the generalized alpha method is unconditionally
    stable and second order accurate with a optimal combination of low
    and high frequency energy dissipation.

    Arguments
    ---------
    M, C, K, F : ndarray
        Mass, Damping and Stiffness matrices defining the structural
        system and the load matrix where each column is the load vector
        f(t) at time t.
    dt : float
        Time increment.
    d0, v0 : 1darray
        Initial displacement and velocity vectors, respectively.
    rho : float
        Determines the numerical damping by the algorithm, a value of 1
        introduces no damping while 0 introduces maximum damping.

    Returns
    -------
    A, V, D : ndarray
        Acceleration, velocity and displacement matrices for the
        solved system.

    Note
    ----
    The generalized alpha algorithm is a family of time integration
    algorithms, including the well known HHT-alpha, WBZ-alpha and
    Newmark method. Check out the generalized_alpha_bf function for
    base form of the algorithm with full access to the time
    integration family.

    Raises
    ------
    ValueError
       If rho is not between 0 and 1 an error is raised to preserve the
       stability and accuracy of the algorithm.
    """
    if (rho<0) or (rho>1):
        raise ValueError("rho should be between 0 and 1 to preserve " \
                         "unconditional stability and second order accuracy")
    alpha_f = rho / (rho + 1)
    alpha_m = (2*rho - 1) / (rho + 1)
    beta = 0.25 * (1-alpha_m+alpha_f)**2
    gamma = 0.5 - alpha_m + alpha_f
    return generalized_alpha_method_bf(M, C, K, F, dt, d0, v0,
                                       alpha_f, alpha_m, beta, gamma)
