"""
CG Solver Tests

Tests the Conjugate Gradient solver using the Field and Laplacian
implementations.

The test uses homogeneous Neumann boundary conditions so that the
operator supplied to CG is linear.
"""

import numpy as np

from Mesh.structuredMesh import StructuredMesh
from Fields.Fields import Field
from Fields.Boundary_Condition import BoundaryCondition
from Fields import Boundary_Condition as BC

from Settings import enums as E

from Operations.laplacian import lap
from Schemes.CG import CG


TOL = 1e-10


def assert_field_close(
    actual: Field,
    expected: Field,
    tol: float = TOL
) -> None:
    assert np.allclose(
        actual._values,
        expected._values,
        atol=tol,
        rtol=tol
    ), (
        f"Got:\n{actual._values}\n"
        f"Expected:\n{expected._values}"
    )


def print_test(name: str) -> None:
    print(f"[PASS] {name}")


def make_1d_mesh() -> StructuredMesh:
    return StructuredMesh(
        [5],
        [(0.0, 1.0)]
    )


def make_zero_gradient_bc(field: Field) -> None:
    field.Boundary_Condition = BoundaryCondition(field)

    field.Boundary_Condition[E.Valid_Sides.LEFT] = BC.zeroGradient()
    field.Boundary_Condition[E.Valid_Sides.RIGHT] = BC.zeroGradient()


def test_CG_scalar() -> None:
    """
    Solve

        A(x) = B

    where

        A(x) = x - alpha * lap(x)

    using homogeneous Neumann boundary conditions.
    """

    mesh = make_1d_mesh()
    alpha = 0.1

    x_exact = Field(
        mesh,
        0,
        0.0,
        name="x_exact",
        unit=""
    )

    # Smooth exact solution with zero derivative at x=0 and x=1.
    for cell in range(mesh.total_cell):
        x = mesh.cells[cell]
        x_exact[0, cell] = 1.0 + np.cos(2.0 * np.pi * x)

    make_zero_gradient_bc(x_exact)

    def A(x: Field) -> Field:
        return x - alpha * lap(x)

    # Construct the RHS from the exact solution.
    B = A(x_exact)

    # Check that the exact solution satisfies A(x_exact) = B.
    exact_residual = B - A(x_exact)

    assert np.allclose(
        exact_residual._values,
        0.0,
        atol=TOL,
        rtol=TOL
    ), (
        "A(x_exact) is not reproduced consistently.\n"
        f"Residual:\n{exact_residual._values}"
    )

    # Check that A is linear.
    x1 = x_exact.copy()
    x2 = x_exact.copy()

    lhs = A(x1 + x2)
    rhs = A(x1) + A(x2)

    linearity_error = np.linalg.norm(
        (lhs - rhs)._values
    )

    assert linearity_error < TOL, (
        "CG operator is not linear.\n"
        f"Linearity error: {linearity_error}"
    )

    # Initial guess.
    x0 = Field(
        mesh,
        0,
        0.0,
        name="x",
        unit=""
    )

    make_zero_gradient_bc(x0)

    # Solve.
    result = CG(
        A,
        x0,
        B,
        tol=1e-10,
        max_iter=100
    )

    # Compare numerical and exact solutions.
    assert_field_close(result, x_exact)

    print_test("scalar CG")


if __name__ == "__main__":
    print("Testing Conjugate Gradient")
    print()

    test_CG_scalar()

    print()
    print("All CG tests passed.")
