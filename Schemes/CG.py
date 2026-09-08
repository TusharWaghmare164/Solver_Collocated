"""
Conjugate Gradient Solver.

Provides:
        - matrix-free Conjugate Gradient iteration
        - Field-based solution and residual handling
        - relative residual convergence criterion
"""

from typing import Callable

import numpy as np

from Fields.Fields import Field


# ========================== Conjugate Gradient ==========================

def CG(
    A: Callable,
    x0: Field,
    B0: Field,
    tol: float = 1e-4,
    max_iter: int = 100
) -> Field:
    """
    Solve the linear system

        A(x) = B

    using the Conjugate Gradient method.

    Parameters
    ----------
    A : Callable
        Linear operator representing A.

        The operator must accept a Field and return a Field:

            A(x) -> Field

    x0 : Field
        Initial guess for the solution.

    B0 : Field
        Right-hand side of the linear system.

    tol : float, optional
        Relative residual tolerance.

    max_iter : int, optional
        Maximum number of CG iterations.

    Returns
    -------
    Field
        Approximate solution of A(x) = B.

    Notes
    -----
    The Conjugate Gradient method requires A to be symmetric
    positive definite with respect to the inner product used
    by the solver.

    The inner products are evaluated over all Field components
    and all cells.
    """

    x = x0
    B = B0

    # Initial residual:
    #
    # r_0 = B - A(x_0)
    #
    r: Field = B - A(x)

    # Initial search direction:
    #
    # p_0 = r_0
    #
    p: Field = r.copy()

    # Initial residual norm:
    #
    # ||r_0||
    #
    r0 = np.linalg.norm(r._values)

    if r0 == 0.0:
        return x

    rtol = 1.0

    for _ in range(max_iter):

        # A p_k
        Ap = A(p)

        # alpha_k =
        #       (r_k, r_k)
        # -----------------
        #       (p_k, A p_k)
        #
        rr = r.inner(r)
        pAp = p.inner(Ap)

        if pAp == 0.0:
            raise ValueError(
                "CG breakdown: p^T A p is zero."
            )

        alpha = rr / pAp

        # x_{k+1} = x_k + alpha_k p_k
        x = x + alpha * p

        # r_{k+1} = r_k - alpha_k A p_k
        r_old = r
        r = r - alpha * Ap

        # Relative residual:
        #
        # ||r_{k+1}||
        # -----------
        #   ||r_0||
        #
        rtol = np.linalg.norm(r._values) / r0

        if rtol <= tol:
            break

        # beta_k =
        #       (r_{k+1}, r_{k+1})
        # ---------------------
        #       (r_k, r_k)
        #
        beta = r.inner(r) / r_old.inner(r_old)

        # p_{k+1} = r_{k+1} + beta_k p_k
        p = r + beta * p

    return x
