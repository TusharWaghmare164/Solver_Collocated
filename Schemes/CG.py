
from typing import Callable

import numpy as np

from Fields.Fields import Field


def CG(
    A: Callable,
    x0: Field,
    B0: Field,
    tol: float = 1e-4,
    max_iter: int = 100
) -> Field:
    """
    Conjugate Gradient solver.

    Solves

        A(x) = B

    using the Conjugate Gradient method.

    Parameters
    ----------
    A : Callable
        Linear operator.

    x0 : Field
        Initial guess for the solution.

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
    # Initial values
    # --------------------------------------------------------------

    x = x0.copy()
    B = B0

    # Initial residual
    r: Field = B - A(x)

    # First search direction
    p: Field = r.copy()

    # Initial residual norm
    rr = r.inner(r)

    if rr == 0.0:
        return x

    r0 = np.sqrt(rr)

    # --------------------------------------------------------------
    # CG iteration
    # --------------------------------------------------------------

    for i in range(max_iter):

        print(f"\nIteration {i}")

        # ----------------------------------------------------------
        # Apply the homogeneous part of A to the search direction.
        #
        # A may contain fixed-value boundary contributions.
        # Therefore:
        #
        #     A_h(p) = A(p) - A(0)
        #
        # where A(0) is the boundary contribution.
        # ----------------------------------------------------------

        zero = p.copy(val_copy=False)
        zero._values[:] = 0.0

        if hasattr(p, "Boundary_Condition"):
            zero.Boundary_Condition = p.Boundary_Condition.copy()

        Ap: Field = A(p) - A(zero)

        # ----------------------------------------------------------
        # alpha
        # ----------------------------------------------------------

        pAp = p.inner(Ap)

        if pAp == 0.0:
            raise ValueError(
                "CG breakdown: p^T A p is zero."
            )

        alpha = rr / pAp

        print(f"  alpha = {alpha}")

        # ----------------------------------------------------------
        # Update solution
        # ----------------------------------------------------------

        x = x + alpha * p

        # ----------------------------------------------------------
        # Update residual
        # ----------------------------------------------------------

        r_old = r

        r = r - alpha * Ap

        # ----------------------------------------------------------
        # Check convergence
        # ----------------------------------------------------------

        rr_new = r.inner(r)

        rtol = np.sqrt(rr_new) / r0

        print(f"  residual = {rtol}")

        if rtol <= tol:
            print("  Converged!")
            break

        # ----------------------------------------------------------
        # beta
        # ----------------------------------------------------------

        beta = rr_new / rr

        print(f"  beta = {beta}")

        # ----------------------------------------------------------
        # New search direction
        # ----------------------------------------------------------

        p = r + beta * p

        # Store new residual norm
        rr = rr_new

    return x
