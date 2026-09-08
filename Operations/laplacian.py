"""
This function is to make laplacian easie
"""

from Fields.Fields import Field
from Fields.Boundary_Condition import BoundaryCondition

from Operations.gradient import grad
from Operations.divergence import div

def lap(field: Field, field2: Field | None = None):

    gradient_field = grad(field, to_face=True)

    if field2 is not None:

        coefficient_faces = field2.to_face()

        for component in range(field.total_component):
            for axis in range(field.dim):

                gradient_field[component][axis] *= (
                    coefficient_faces[component][axis]
                )

    assert isinstance(gradient_field, list)

    laplacian_field = div(
        field,
        gradient_field
    )

    return laplacian_field
