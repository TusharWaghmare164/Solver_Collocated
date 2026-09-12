"""
Checkerboard Pressure Test

Checks whether a collocated pressure field with a checkerboard pattern
is visible through the current face interpolation and gradient operators.
"""

import numpy as np

from Mesh.structuredMesh import StructuredMesh
from Fields.Field_maker import Scalar
from Operations.gradient import grad


TOL = 1e-12


def make_mesh():
    return StructuredMesh(
        [8, 8],
        [
            (0.0, 1.0),
            (0.0, 1.0)
        ]
    )


def make_checkerboard_pressure(mesh):
    p = Scalar(
        mesh,
        0.0,
        name="Pressure",
        unit="Pa"
    )

    nx, ny = mesh.partitions

    for j in range(ny):
        for i in range(nx):
            cell = j * nx + i
            p[0, cell] = (-1.0) ** (i + j)

    return p


def test_checkerboard_pattern(field):

    nx, ny = field.mesh.partitions

    expected = np.zeros(field.mesh.total_cell)

    for j in range(ny):
        for i in range(nx):
            cell = j * nx + i
            expected[cell] = (-1.0) ** (i + j)

    assert np.allclose(
        field._values[0],
        expected,
        atol=TOL
    )


def test_face_interpolation(field):

    face_values = field.to_face()

    max_face_value = 0.0

    for axis in range(field.mesh.dim):

        neighbours = field.mesh.face_neighbour_cells[axis]

        for face, (minus_cell, plus_cell) in enumerate(neighbours):

            if minus_cell == -1 or plus_cell == -1:
                continue

            value = abs(face_values[0][axis][face])

            max_face_value = max(
                max_face_value,
                value
            )

    print(
        f"Maximum interior face pressure: "
        f"{max_face_value:.6e}"
    )

    assert max_face_value < TOL


def test_face_gradient(field):

    gradient = grad(
        field,
        to_face=True
    )

    max_gradient = 0.0

    for component in gradient:

        for axis in range(field.mesh.dim):

            neighbours = field.mesh.face_neighbour_cells[axis]

            for face, (minus_cell, plus_cell) in enumerate(neighbours):

                if minus_cell == -1 or plus_cell == -1:
                    continue

                value = abs(
                    component[axis][face]
                )

                max_gradient = max(
                    max_gradient,
                    value
                )

    print(
        f"Maximum interior face pressure gradient: "
        f"{max_gradient:.6e}"
    )

    assert max_gradient > TOL


def print_pressure(field):

    nx, ny = field.mesh.partitions

    print("\nCell-centered pressure:\n")

    for j in reversed(range(ny)):

        row = []

        for i in range(nx):

            cell = j * nx + i

            row.append(
                f"{field[0, cell]:5.1f}"
            )

        print(" ".join(row))


def run_tests():

    print("=" * 60)
    print("Checkerboard Pressure Test")
    print("=" * 60)

    mesh = make_mesh()

    pressure = make_checkerboard_pressure(mesh)

    print_pressure(pressure)

    print("\nTesting checkerboard pattern...")

    test_checkerboard_pattern(pressure)

    print("[PASS] Checkerboard pattern")

    print("\nTesting face interpolation...")

    test_face_interpolation(pressure)

    print(
        "[PASS] Interior face interpolation "
        "collapses to zero"
    )

    print("\nTesting face pressure gradient...")

    test_face_gradient(pressure)

    print(
        "[PASS] Gradient detects "
        "checkerboard pressure"
    )

    print("\n" + "=" * 60)
    print("All checkerboard tests passed.")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
