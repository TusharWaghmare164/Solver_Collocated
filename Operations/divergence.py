from copy import deepcopy

from Fields.Fields import Field
from gradient import grad

def div(field:Field):

    divergence_field = Field(field.mesh, 0, 0.0, name=f"div_{field.name}", unit=f"{field.unit}/L")
    divergence_field.Boundary_Condition = deepcopy(field.Boundary_Condition)

    gradient_field = grad(field)

    for i in range(field.dim):
        divergence_field._values[0] += gradient_field[i * field.dim + i]

    return divergence_field
