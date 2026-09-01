import numpy as np
from Fields.Fields import Field

def gradient(field: Field):

    """
    This function will calculate the gradient of a field and return it as a new Field object.
    The gradient is calculated using face_values of field and stored as new field.
    """

    mesh = field.mesh

    # Get the face values of the field
    face_values = field.to_face()

    # Initialize a new Field object for the gradient
    gradient_field = Field(mesh, field.field_order + 1, 0.0, name=f"grad_{field.name}", unit=f"{field.unit}/L")

    # Calculate the gradient for each component of the field
    for component in range(field.field_order + 1):
        component_faces = face_values[component]

        # Calculate the gradient using central differences
        for axis in range(mesh.dim):
            # Get the face values in the current axis direction
            face_values_axis = component_faces[axis]

            # Calculate the gradient using central differences
            gradient_component = np.zeros_like(face_values_axis)

            # Central difference cell
            for cell in range(mesh.total_cell):

                # Get the indices of the neighboring cells
                plus_cell = mesh.face_neighbour_cells[axis][cell][1]
                minus_cell = mesh.face_neighbour_cells[axis][cell][0]

                # Calculate the gradient using central differences
                gradient_component[cell] = (face_values_axis[plus_cell] - face_values_axis[minus_cell]) / (mesh.distances[axis][cell] + mesh.distances[axis][minus_cell])

            # Store the gradient component in the new field
            gradient_field._values[component] = gradient_component

    return gradient_field
