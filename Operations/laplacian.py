"""
This function is to make Laplacian easier.
"""

from Fields.Fields import Field

from Operations.gradient import grad
from Operations.divergence import div


def lap(
    field: Field,
    field2: Field | int | float | None = None
):
    """Calculate the Laplacian of a field."""

    gradient_field = grad(
        field,
        to_face=True
    )

    if field2 is not None:

        if isinstance(field2, Field):

            if field2.field_order != 0:
                raise ValueError(
                    "Laplacian coefficient must be a scalar Field."
                )

            coefficient_faces = field2.to_face()

            # Multiply each face gradient by the corresponding
            # spatially varying coefficient.
            for component in range(field.total_component):
                for axis in range(field.dim):
                    gradient_field[component][axis] *= (
                        coefficient_faces[0][axis]
                    )

        elif isinstance(field2, (int, float)):

            # Multiply the gradient by a constant coefficient.
            for component in range(field.total_component):
                for axis in range(field.dim):
                    gradient_field[component][axis] *= field2

        else:
            raise TypeError(
                "field2 must be a Field, int, float, or None."
            )

    return div(
        field,
        gradient_field
    )
