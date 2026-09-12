
from typing import Callable

import numpy as np

from Fields.Fields import Field


def BiCGSTAB(
    A: Callable,
    x0: Field,
    B0: Field,
    tol: float = 1e-4,
    max_iter: int = 100
) -> Field:
    """
    BiCGSTAB solver.

    Solves

        A(x) = B

    using the Bi-Conjugate Gradient Stabilized method.

    Parameters
    ----------
    A : Callable
        Linear operator.

    x0 : Field
        Initial guess.

    B0 : Field
        Right-hand side.

    tol : float, optional
        Relative residual tolerance.

    max_iter : int, optional
        Maximum number of iterations.

    Returns
    -------
    Field
        Approximate solution.
    """

    # --------------------------------------------------------------
    # Initial solution and RHS
    # --------------------------------------------------------------

    x = x0.copy()
    B = B0

    # --------------------------------------------------------------
    # Initial residual
    #
    # r = B - A(x)
    # --------------------------------------------------------------

    r: Field = B - A(x)

    # Choose the initial shadow residual.
    #
    # BiCGSTAB requires a fixed vector r_hat such that
    #
    #     r_hat^T r != 0
    #
    r_hat: Field = r.copy()

    # --------------------------------------------------------------
    # Initial residual norm
    # --------------------------------------------------------------

    r0_norm = np.sqrt(r.inner(r))

    if r0_norm == 0.0:
        return x

    # --------------------------------------------------------------
    # Initial variables
    # --------------------------------------------------------------

    rho_old = 1.0
    alpha = 1.0
    omega = 1.0

    p = r.copy()
    v = r.copy()

    # --------------------------------------------------------------
    # BiCGSTAB iteration
    # --------------------------------------------------------------

    for i in range(max_iter):

        print(f"\nIteration {i}")

        # ----------------------------------------------------------
        # rho = r_hat^T r
        # ----------------------------------------------------------

        rho_new = r_hat.inner(r)

        if abs(rho_new) < 1e-30:
            raise ValueError(
                "BiCGSTAB breakdown: r_hat^T r is zero."
            )

        # ----------------------------------------------------------
        # First iteration
        # ----------------------------------------------------------

        if i == 0:
            p = r.copy()

        else:
            beta = (
                (rho_new / rho_old)
                * (alpha / omega)
            )

            p = r + beta * (p - omega * v)

        # ----------------------------------------------------------
        # Apply operator
        # ----------------------------------------------------------

        v = A(p)

        # ----------------------------------------------------------
        # alpha
        # ----------------------------------------------------------

        r_hat_v = r_hat.inner(v)

        if abs(r_hat_v) < 1e-30:
            raise ValueError(
                "BiCGSTAB breakdown: r_hat^T A p is zero."
            )

        alpha = rho_new / r_hat_v

        # ----------------------------------------------------------
        # Intermediate residual
        #
        # s = r - alpha*v
        # ----------------------------------------------------------

        s = r - alpha * v

        # ----------------------------------------------------------
        # Check whether s is already small
        # ----------------------------------------------------------

        s_norm = np.sqrt(s.inner(s))
        s_rtol = s_norm / r0_norm

        print(f"  alpha = {alpha}")
        print(f"  intermediate residual = {s_rtol}")

        if s_rtol <= tol:

            x = x + alpha * p

            print("  Converged!")

            break

        # ----------------------------------------------------------
        # Apply operator to s
        # ----------------------------------------------------------

        t = A(s)

        # ----------------------------------------------------------
        # omega
        #
        # omega = (t^T s) / (t^T t)
        # ----------------------------------------------------------

        tt = t.inner(t)

        if abs(tt) < 1e-30:
            raise ValueError(
                "BiCGSTAB breakdown: A(s)^T A(s) is zero."
            )

        omega = t.inner(s) / tt

        # ----------------------------------------------------------
        # Update solution
        #
        # x = x + alpha*p + omega*s
        # ----------------------------------------------------------

        x = x + alpha * p + omega * s

        # ----------------------------------------------------------
        # Update residual
        #
        # r = s - omega*t
        # ----------------------------------------------------------

        r = s - omega * t

        # ----------------------------------------------------------
        # Check convergence
        # ----------------------------------------------------------

        r_norm = np.sqrt(r.inner(r))
        rtol = r_norm / r0_norm

        print(f"  omega = {omega}")
        print(f"  residual = {rtol}")

        if rtol <= tol:

            print("  Converged!")

            break

        if abs(omega) < 1e-30:
            raise ValueError(
                "BiCGSTAB breakdown: omega is zero."
            )

        # ----------------------------------------------------------
        # Prepare next iteration
        # ----------------------------------------------------------

        rho_old = rho_new

    return x
