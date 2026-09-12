import numpy as np
from copy import deepcopy
from typing import Sequence

from Fields.Fields import Field
import Settings.enums as E
from Utilities.components import make_component_labels


def grad(field: Field, face_values: None|Sequence = None, to_face:bool = False):

    """
    This function will calculate the gradient of a field and return it as a new Field object.
    The gradient is calculated using face_values of field and stored as new field.
    """

    mesh = field.mesh

    # Get the face values of the field
    face_values = field.to_face() if face_values is None else face_values

    if not to_face:

        # Initialize a new Field object for the gradient
        gradient_field = Field(mesh, field.field_order + 1, 0.0, name=f"grad_{field.name}", unit=f"{field.unit}/L")
        # gradient_field.Boundary_Condition = deepcopy(field.Boundary_Condition)

        # Calculate the gradient for each component of the field
        for component in range(field.total_component):

            # Calculate the gradient using central differences
            for axis in range(mesh.dim):

                # Calculate the gradient using central differences
                gradient_component = np.zeros(mesh.total_cell)

                # Central difference cell
                for cell in range(mesh.total_cell):

                    faces = mesh.cell_faces[cell]

                    minus_face = faces[2 * axis]
                    plus_face = faces[2 * axis + 1]

                    gradient_component[cell] = (
                        face_values[component][axis][plus_face]
                        - face_values[component][axis][minus_face]
                    ) / mesh.distances[axis]

                # Store the gradient component in the new field
                gradient_component_index = component * mesh.dim + axis
                gradient_field._values[gradient_component_index] = gradient_component

        assert isinstance(gradient_field, Field)

        return gradient_field

    else:

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
                    return (internal_value - BC.bc_value) / distance
                else:
                    return (BC.bc_value - internal_value) / distance

            raise ValueError(
                f"Unsupported boundary condition type: {BC.bc_type}"
            )

        Sides = {
            0: [E.Valid_Sides.LEFT, E.Valid_Sides.RIGHT],
            1: [E.Valid_Sides.BOTTOM, E.Valid_Sides.TOP],
            2: [E.Valid_Sides.BACK, E.Valid_Sides.FRONT]
        }

        mesh = field.mesh

        components = (
            ["scalar"]
            if field.field_order == 0
            else make_component_labels(
                field.dim,
                field.field_order
            )
        )

        face_values = []

        for component in range(field.total_component):

            component_faces = []

            for axe in range(mesh.dim):

                total_faces = mesh.total_faces[axe]

                face_value = np.zeros(total_faces)

                for idx in range(total_faces):

                    neighbours = mesh.face_neighbour_cells[axe][idx]

                    # ---------------- Interior ----------------

                    if -1 not in neighbours:

                        c0 = neighbours[0]
                        c1 = neighbours[1]

                        face_value[idx] = (
                            field[component, c1]
                            -
                            field[component, c0]
                        ) / mesh.distances[axe]

                    # ---------------- Boundary ----------------

                    else:

                        boundary_index = np.where(
                            neighbours == -1
                        )[0][0]

                        internal_cell = neighbours[
                            1 - boundary_index
                        ]

                        side = Sides[axe][boundary_index]

                        # Get correct BC
                        if field.field_order == 0:

                            bc = field.Boundary_Condition[side]

                        else:

                            component_name = components[component]

                            bc = field.Boundary_Condition[
                                component_name,
                                side
                            ]

                        face_value[idx] = boundary_value(
                            BC=bc,
                            internal_value=field[
                                component,
                                internal_cell
                            ], #pyright: ignore
                            opposite=(boundary_index == 0),
                            distance=mesh.distances[axe] / 2
                        )

                component_faces.append(face_value)

            face_values.append(component_faces)

            assert isinstance(face_value, np.ndarray)

        return face_values
