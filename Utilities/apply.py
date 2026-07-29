import numpy as np

from Fields.Base_field import BaseField

def apply(func, *fields):

    mesh = fields[0].mesh

    mesh = fields[0].mesh

    for field in fields:

        if field.mesh is not mesh:
            raise ValueError(
                "All fields must belong to the same mesh"
            )

    result = BaseField(mesh)

    inputs = [field.values for field in fields]

    result.values = func(*inputs)

    return result
