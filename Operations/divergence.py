from Fields.Fields import Field

def div(
    field: Field,
    grad_values: None | list = None
):
    """Calculate the divergence of a field."""

    mesh = field.mesh

    if grad_values is None:
        if field.field_order == 0:
            raise ValueError(
                "Divergence is not defined for a scalar field."
            )

        face_values = field.to_face()

        divergence_field = Field(
            mesh,
            field.field_order - 1,
            0.0,
            name=f"div_{field.name}",
            unit=f"{field.unit}/L"
        )

        for axis in range(mesh.dim):
            minus_faces = mesh.cell_faces[:, 2 * axis]
            plus_faces = mesh.cell_faces[:, 2 * axis + 1]

            for component in range(divergence_field.total_component):
                field_component = component * mesh.dim + axis

                divergence_field._values[component] += (
                    face_values[field_component][axis][plus_faces]
                    - face_values[field_component][axis][minus_faces]
                ) / mesh.distances[axis]

    else:
        face_values = grad_values

        divergence_field = Field(
            mesh,
            field.field_order,
            0.0,
            name=f"lap_{field.name}",
            unit=f"{field.unit}/L^2"
        )

        for axis in range(mesh.dim):
            minus_faces = mesh.cell_faces[:, 2 * axis]
            plus_faces = mesh.cell_faces[:, 2 * axis + 1]

            # grad_values already has the gradient component
            # for this spatial direction.
            gradient_component = axis

            divergence_field._values[0] += (
                face_values[gradient_component][axis][plus_faces]
                - face_values[gradient_component][axis][minus_faces]
            ) / mesh.distances[axis]

    divergence_field.Boundary_Condition = (
        field.Boundary_Condition
    )

    return divergence_field
