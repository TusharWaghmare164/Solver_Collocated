
import numpy as np

from Fields.Fields import Field
from Fields.Face_field import Face_values


def dot(field1: Field | Face_values,
        field2: Field | Face_values):

    if type(field1) is not type(field2):
        raise ValueError(
            "Both operands must be of the same type"
        )

    if field1.mesh is not field2.mesh:
        raise ValueError(
            "Both fields must share the same mesh"
        )

    if field1.order == 0 or field2.order == 0:
        raise ValueError(
            "Dot product requires both operands "
            "to have nonzero order"
        )

    dim = field1.dim
    mesh = field1.mesh

    order1 = field1.order
    order2 = field2.order
    result_order = order1 + order2 - 2

    result = type(field1)(mesh, result_order)

    def contract(a, b):
        a = np.asarray(a).reshape((dim,) * order1)
        b = np.asarray(b).reshape((dim,) * order2)

        return np.tensordot(a, b, axes=([-1], [0])).reshape(-1)

    # Face_values · Face_values
    if isinstance(field1, Face_values):

        for d in range(dim):
            for f in range(mesh.total_faces[d]):

                a = [field1[c][d][f]
                     for c in range(field1.total_component)]

                b = [field2[c][d][f]
                     for c in range(field2.total_component)]

                values = contract(a, b)

                for c, value in enumerate(values):
                    result[c][d][f] = value

        return result

    # Field · Field
    for cell in range(mesh.total_cell):

        a = [field1[c][cell]
             for c in range(field1.total_component)]

        b = [field2[c][cell]
             for c in range(field2.total_component)]

        values = contract(a, b)

        for c, value in enumerate(values):
            result[c][cell] = value

    # Also calculate face values when both are available
    if (
        isinstance(field1._face_value, Face_values)
        and isinstance(field2._face_value, Face_values)
    ):
        result._face_value = dot(
            field1._face_value,
            field2._face_value
        )

    return result
