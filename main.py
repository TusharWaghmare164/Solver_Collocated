import numpy as np

from Mesh.structuredMesh import StructuredMesh
from Fields.Field_maker import Scalar, Vector

from Fields.Boundary_Condition import BoundaryCondition
import Fields.Boundary_Condition as BC
import Settings.enums as E

from Operations.grad import grad
from Operations.div import div
from Operations.lap import lap
from Operations.dot import dot

from Solver.dt import implicite


def main():

    # ============================================================
    # Problem
    # ============================================================

    length = 1.0
    partition = [100,100]

    ALPHA = 0.02

    t_final = 0.1

    # ============================================================
    # Mesh
    # ============================================================

    mesh = StructuredMesh(
        partition,
        # [(0.0, length),(0.0, length),(0.0, length)]
        [(0.0, length),(0.0, length)]
    )

    # ============================================================
    # Temperature
    # ============================================================

    T = Scalar(
        mesh,
        0.0,
        name="Temperature",
        unit="degC"
    )

    U = Vector(
        mesh,
        0,
        "velocity",
        "m/s"
    )

    U[0][:] = 0.1
    U[1][:] = 0.05
    # U[1][:] = 0.02

    # ============================================================
    # Boundary conditions
    # ============================================================

    bc = BoundaryCondition(T.dim, T.order)

    bc[E.Valid_Sides.LEFT] = BC.fixedValue(100)
    bc[E.Valid_Sides.RIGHT] = BC.fixedValue(0)
    bc[E.Valid_Sides.TOP] = BC.fixedValue(0)
    bc[E.Valid_Sides.BOTTOM] = BC.fixedValue(100)
    # bc[E.Valid_Sides.FRONT] = BC.Gradient(0)
    # bc[E.Valid_Sides.BACK] = BC.fixedValue(300)

    T.Boundary_condition = bc

    def equation(x):
        # return (ALPHA * lap(x))
        return (ALPHA * lap(x)) - dot(U,grad(x))

    # Solving the Equation

    implicite(
        equation,
        T,
        10,
        time_steps=0.1
    )


if __name__ == "__main__":
    main()
