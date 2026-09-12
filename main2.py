import numpy as np

from Mesh.structuredMesh import StructuredMesh
from Fields.Field_maker import Scalar

from Fields.Boundary_Condition import BoundaryCondition
import Fields.Boundary_Condition as BC
import Settings.enums as E

from Operations.laplacian import lap
from Schemes.CG import CG


def main():

    # ============================================================
    # Problem parameters
    # ============================================================

    length = 10.0
    partition = [10]

    gamma = 0.1

    T_left = 100.0
    T_right = 300.0

    # ============================================================
    # Mesh
    # ============================================================

    mesh = StructuredMesh(
        partition,
        [(0.0, length)]
    )

    # ============================================================
    # Temperature field
    # ============================================================

    T = Scalar(
        mesh,
        50.0,
        name="Temperature",
        unit="degC"
    )

    # ============================================================
    # Boundary conditions
    # ============================================================

    bc = BoundaryCondition(T)

    bc[E.Valid_Sides.LEFT] = BC.fixedValue(T_left)
    bc[E.Valid_Sides.RIGHT] = BC.fixedValue(T_right)

    # bc[E.Valid_Sides.RIGHT] = BC.Gradient(2)
    # bc[E.Valid_Sides.LEFT] = BC.Gradient(3)

    T.Boundary_Condition = bc

    # ============================================================
    # Diffusion equation
    #
    #       lap(T) = 0
    #
    # Gamma is constant, so it cancels from the steady
    # equation. We keep it here because the actual diffusion
    # operator should eventually be represented explicitly.
    # ============================================================

    def A(x):

        return lap(x)

    # ============================================================
    # Initial guess
    # ============================================================

    T_initial = T.copy()

    # ============================================================
    # Solve
    # ============================================================

    # NOTE:
    # This particular equation has Dirichlet BCs, so the
    # operator/BC treatment must be linear for CG to be valid.

    B = T.copy()
    B._values[:] = 0.0

    T_solution = CG(
        A,
        T_initial,
        B,
        tol=1e-10,
        max_iter=1000
    )

    # ============================================================
    # Analytical solution
    # ============================================================

    x = mesh.cells[:, 0]

    analytical = T_solution.copy()

    analytical[0] = T_left + (
        (T_right - T_left) / length
    ) * x

    # ============================================================
    # Print result
    # ============================================================

    print()
    print("=" * 60)
    print("1D STEADY DIFFUSION TEST")
    print("=" * 60)

    print(f"Length       : {length}")
    print(f"Cells        : {partition[0]}")
    print(f"Gamma        : {gamma}")
    print(f"Left T       : {T_left}")
    print(f"Right T      : {T_right}")

    print()
    print("Temperature:")
    print(T_solution._values)

    print()
    print("Analytical:")
    print(analytical._values)

    error = np.max(
        np.abs(
            T_solution._values -
            analytical._values
        )
    )

    print()
    print(f"Maximum error: {error:.6e}")

    if error < 1e-10:
        print("[PASS] Diffusion test")
    else:
        print("[FAIL] Diffusion test")

    print("=" * 60)

    return T_solution


if __name__ == "__main__":
    main()
