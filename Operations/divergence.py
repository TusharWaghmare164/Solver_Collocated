from copy import deepcopy

from Fields.Fields import Field
from Operations.gradient import grad


def div(field: Field, face_values: None | list = None):

    mesh = field.mesh

    if face_values is None:

        if field.field_order == 0:
            raise ValueError(
                "Divergence is not defined for a scalar field."
            )

        divergence_field = Field(
            field.mesh,
            field.field_order - 1,
            0.0,
            name=f"div_{field.name}",
            unit=f"{field.unit}/L"
        )

        face_values = field.to_face()

    else:
        divergence_field = Field(
            field.mesh,
            field.field_order,
            0.0,
            name=f"div_{field.name}",
            unit=f"{field.unit}/L"
        )

    for component in range(divergence_field.total_component):

        for cell in range(mesh.total_cell):

            faces = mesh.cell_faces[cell]

            for axis in range(mesh.dim):

                minus_face = faces[2 * axis]
                plus_face = faces[2 * axis + 1]

                divergence_field[component, cell] += (
                    face_values[component][axis][plus_face]
                    - face_values[component][axis][minus_face]
                ) / mesh.distances[axis]

    # for component in range(divergence_field.total_component):

    #     for axis in range(field.dim):

    #         field_component = component * field.dim + axis
    #         gradient_component = (
    #             field_component * field.dim + axis
    #         )

    #         divergence_field[component] += (
    #             gradient_field[gradient_component]
    #         )

    divergence_field.Boundary_Condition = field.Boundary_Condition

    return divergence_field
