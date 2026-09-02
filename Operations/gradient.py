import numpy as np
from copy import deepcopy

from Fields.Fields import Field

def grad(field: Field):

    """
    This function will calculate the gradient of a field and return it as a new Field object.
    The gradient is calculated using face_values of field and stored as new field.
    """

    mesh = field.mesh

    # Get the face values of the field
    face_values = field.to_face()

    # Initialize a new Field object for the gradient
    gradient_field = Field(mesh, field.field_order + 1, 0.0, name=f"grad_{field.name}", unit=f"{field.unit}/L")
    gradient_field.Boundary_Condition = deepcopy(field.Boundary_Condition)

    # Calculate the gradient for each component of the field
    for component in range(field.total_component):

        # Calculate the gradient using central differences
        for axis in range(mesh.dim):

            # Calculate the gradient using central differences
            gradient_component = np.zeros(mesh.total_cell)

            # Central difference cell
            for cell in range(mesh.total_cell):

                faces = mesh.cell_faces[cell]

                plus_face = faces[2 * axis]
                minus_face = faces[2 * axis + 1]

                gradient_component[cell] = (
                    face_values[component][axis][plus_face]
                    - face_values[component][axis][minus_face]
                ) / mesh.distances[axis]

            # Store the gradient component in the new field
            gradient_component_index = component * mesh.dim + axis
            gradient_field._values[gradient_component_index] = gradient_component

    return gradient_field
