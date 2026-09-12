import numpy as np
from copy import deepcopy
from typing import Sequence

from Fields.Fields import Field
import Settings.enums as E
from Utilities.components import make_component_labels


def grad(
    field: Field,
    face_values: None | Sequence = None,
    to_face: bool = False
):
    """
    Calculate the gradient of a field.

    If to_face is False, the gradient is returned at cell centres.
    If to_face is True, the gradient is returned at face centres.
    """

    mesh = field.mesh

    if not to_face:

        face_values = (
            field.to_face()
            if face_values is None
            else face_values
        )

        gradient_field = Field(
            mesh,
            field.field_order + 1,
            0.0,
            name=f"grad_{field.name}",
            unit=f"{field.unit}/L"
        )

        # Calculate the gradient for every cell using the two
        # faces surrounding each cell along the given axis.
        for axis in range(mesh.dim):

            minus_faces = mesh.cell_faces[:, 2 * axis]
            plus_faces = mesh.cell_faces[:, 2 * axis + 1]

            for component in range(field.total_component):

                gradient_component_index = (
                    component * mesh.dim + axis
                )

                gradient_field._values[
                    gradient_component_index
                ] = (
                    face_values[component][axis][plus_faces]
                    -
                    face_values[component][axis][minus_faces]
                ) / mesh.distances[axis]

        return gradient_field

    from Fields.Boundary_Condition import BCEntry

    def boundary_value(
        BC: BCEntry,
        internal_value: float,
        opposite: bool,
        distance: float
    ) -> float:

        if BC.bc_type == E.BoundaryType.GRADIENT:
            return BC.bc_value

        elif BC.bc_type == E.BoundaryType.FIXED_VALUE:
            if opposite:
                return (
                    internal_value - BC.bc_value
                ) / distance
            else:
                return (
                    BC.bc_value - internal_value
                ) / distance

        raise ValueError(
            f"Unsupported boundary condition type: {BC.bc_type}"
        )

    sides = {
        0: [E.Valid_Sides.LEFT, E.Valid_Sides.RIGHT],
        1: [E.Valid_Sides.BOTTOM, E.Valid_Sides.TOP],
        2: [E.Valid_Sides.BACK, E.Valid_Sides.FRONT]
    }

    components = (
        ["scalar"]
        if field.field_order == 0
        else make_component_labels(
            field.dim,
            field.field_order
        )
    )

    gradient_components = field.total_component * mesh.dim

    face_values = [
        [
            np.zeros(mesh.total_faces[axis], dtype=float)
            for axis in range(mesh.dim)
        ]
        for _ in range(gradient_components)
    ]

    for axis in range(mesh.dim):

        inner_faces = mesh.inner_faces[axis]
        inner_cells = mesh.inner_cells[axis]

        # Interior face gradients can be calculated for all faces
        # at once from their two neighbouring cells.
        if inner_faces.size > 0:

            c0 = inner_cells[:, 0]
            c1 = inner_cells[:, 1]

            for component in range(field.total_component):

                face_values[component][axis][inner_faces] = (
                    field._values[component, c1]
                    -
                    field._values[component, c0]
                ) / mesh.distances[axis]

        boundary_faces = mesh.boundary_faces[axis]
        boundary_cells = mesh.boundary_cells[axis]

        # Boundary gradients depend on the boundary condition,
        # so only the boundary faces need to be processed individually.
        for idx in range(boundary_faces.size):

            face = boundary_faces[idx]
            neighbours = boundary_cells[idx]

            boundary_index = np.where(
                neighbours == -1
            )[0][0]

            internal_cell = neighbours[
                1 - boundary_index
            ]

            side = sides[axis][boundary_index]

            for component in range(field.total_component):

                if field.field_order == 0:
                    bc = field.Boundary_Condition[side]
                else:
                    component_name = components[component]

                    bc = field.Boundary_Condition[
                        component_name,
                        side
                    ]

                face_values[component][axis][face] = boundary_value(
                    BC=bc,
                    internal_value=field._values[
                        component,
                        internal_cell
                    ],
                    opposite=(boundary_index == 0),
                    distance=mesh.distances[axis] / 2
                )

    return face_values
