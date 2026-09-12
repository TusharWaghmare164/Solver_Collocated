from typing import Callable
import numpy as np

from Fields.Fields import Field


def Richardson(
    A: Callable,
    b: Field,
    max_iter: int = 5
) -> Field:

    # Start with zero solution
    x = b.copy(val_copy=False)
    x[:] = 0.0

    # Calculate initial residual
    r = b - A(x)
    r_norm = np.sqrt(r.inner(r))

    if r_norm == 0.0:
        return x

    for i in range(max_iter):

        # Calculate A(r)
        Ar = A(r)

        # Calculate Richardson relaxation parameter
        Ar_norm_squared = Ar.inner(Ar)

        if abs(Ar_norm_squared) < 1e-30:
            break

        omega = r.inner(Ar) / Ar_norm_squared

        # Save previous solution
        x_previous = x.copy()

        # Richardson update
        x = x + omega * r

        # Calculate new residual
        r_new = b - A(x)
        r_new_norm = np.sqrt(r_new.inner(r_new))

        print(
            f"    Richardson iteration {i}: "
            f"omega = {omega}, "
            f"residual = {r_new_norm}"
        )

        # Stop if residual increases
        if r_new_norm > r_norm:
            print(
                "    Richardson residual increased. "
                "Using previous solution."
            )
            return x_previous

        # Accept new solution
        r = r_new
        r_norm = r_new_norm

        # Stop if converged
        if r_norm == 0.0:
            break

    return x
