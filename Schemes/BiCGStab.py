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

    # Initial solution and RHS
    x = x0.copy()
    B = B0

    # Calculate initial residual
    r: Field = B - A(x)

    # Set shadow residual
    r_hat: Field = r.copy()

    # Calculate initial residual norm
    r0_norm = np.sqrt(r.inner(r))

    if r0_norm == 0.0:
        return x

    # Initialize BiCGSTAB variables
    rho_old = 1.0
    alpha = 1.0
    omega = 1.0

    p = r.copy()
    v = r.copy()

    for i in range(max_iter):

        print(f"\nIteration {i}")

        # Calculate rho
        rho_new = r_hat.inner(r)

        if abs(rho_new) < 1e-30:
            raise ValueError(
                "BiCGSTAB breakdown: r_hat^T r is zero."
            )

        # Calculate search direction
        if i == 0:
            p = r.copy()
        else:
            beta = (
                (rho_new / rho_old)
                * (alpha / omega)
            )

            p = r + beta * (p - omega * v)

        # Apply operator
        v = A(p)

        # Calculate alpha
        r_hat_v = r_hat.inner(v)

        if abs(r_hat_v) < 1e-30:
            raise ValueError(
                "BiCGSTAB breakdown: r_hat^T A(p) is zero."
            )

        alpha = rho_new / r_hat_v

        # Calculate intermediate residual
        s = r - alpha * v

        s_norm = np.sqrt(s.inner(s))
        s_rtol = s_norm / r0_norm

        print(f"  alpha = {alpha}")
        print(f"  intermediate residual = {s_rtol}")

        # Check intermediate convergence
        if s_rtol <= tol:
            x = x + alpha * p
            print("  Converged!")
            break

        # Apply operator
        t = A(s)

        # Calculate BiCGSTAB omega
        tt = t.inner(t)

        if abs(tt) < 1e-30:
            raise ValueError(
                "BiCGSTAB breakdown: A(s)^T A(s) is zero."
            )

        omega = t.inner(s) / tt

        # Update solution
        x = x + alpha * p + omega * s

        # Update residual
        r = s - omega * t

        # Calculate relative residual
        r_norm = np.sqrt(r.inner(r))
        rtol = r_norm / r0_norm

        print(f"  omega = {omega}")
        print(f"  residual = {rtol}")

        # Check convergence
        if rtol <= tol:
            print("  Converged!")
            break

        # Check BiCGSTAB breakdown
        if abs(omega) < 1e-30:
            raise ValueError(
                "BiCGSTAB breakdown: omega is zero."
            )

        # Prepare next iteration
        rho_old = rho_new

    return x
