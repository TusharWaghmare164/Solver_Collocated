import numpy as np
from typing import overload

from Fields.Fields import Field
from Fields.Face_field import Face_values
from Fields.Boundary_Condition import BCEntry
from Settings import enums as E

@overload
def boundary_value(
    BC: BCEntry,
    internal_value: np.ndarray,
    opposite: bool,
    distance: float
) -> np.ndarray: ...

@overload
def boundary_value(
    BC: BCEntry,
    internal_value: float,
    opposite: bool,
    distance: float
) -> float: ...


def boundary_value(
    BC: BCEntry,
    internal_value: float | np.ndarray,
    opposite: bool,
    distance: float
) -> float | np.ndarray:

    if BC.bc_type == E.BoundaryType.FIXED_VALUE:
        return BC.bc_value

    elif BC.bc_type == E.BoundaryType.GRADIENT:
        if opposite:
            return internal_value - (BC.bc_value * distance)
        else:
            return internal_value + (BC.bc_value * distance)

    raise ValueError(
        f"Unsupported boundary condition type: {BC.bc_type}"
    )


def to_face(field: Field):

    sides = {
    0: [E.Valid_Sides.LEFT, E.Valid_Sides.RIGHT],
    1: [E.Valid_Sides.BOTTOM, E.Valid_Sides.TOP],
    2: [E.Valid_Sides.BACK, E.Valid_Sides.FRONT]
    }

    order = field.order
    mesh = field.mesh
    dim = mesh.dim

    result = Face_values(mesh, order)

    face_neighbour = mesh.face_neighbour_cells

    for d in range(dim):

        assert field.Boundary_condition is not None

        neighbour = face_neighbour[d]
        minus = neighbour[:, 0]
        plus = neighbour[:, 1]

        inner_face = mesh.inner_faces[d]
        inner_minus = minus[inner_face]
        inner_plus = plus[inner_face]

        boundary_face = mesh.boundary_faces[d]
        minus_boundary = boundary_face[0]
        plus_boundary = boundary_face[1]

        bc_minus = field.Boundary_condition[sides[d][0]]
        bc_plus = field.Boundary_condition[sides[d][1]]

        for c in range(field.total_component):

            result[c][d][inner_face] = (
                field[c][inner_minus] + field[c][inner_plus]
            )/2

            result[c][d][minus_boundary] = boundary_value(
                BC= bc_minus,
                internal_value= field[c][plus[minus_boundary]],     # pyright: ignore
                opposite= True,
                distance= mesh.distances[d] / 2
            )

            result[c][d][plus_boundary] = boundary_value(
                BC= bc_plus,
                internal_value= field[c][minus[plus_boundary]],     # pyright: ignore
                opposite= False,
                distance= mesh.distances[d] / 2
            )

    return result


def to_cell(field: Field, face_value: Face_values | None = None):

    dim = field.dim
    mesh = field.mesh

    if face_value is None:
        face_value = field._face_value

    assert face_value is not None

    cell_neigh = mesh.cell_faces

    for c in range(field.total_component):

        val = 0

        for d in range(dim):

            minus = cell_neigh[:,d,0]
            plus = cell_neigh[:,d,1]

            val += face_value[c][d][minus] + face_value[c][d][plus]

        field[c] = val / (2*dim)

    return field
