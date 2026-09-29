from Fields.Fields import Field
from Fields.Face_field import Face_values


def mul(field1: Field | Face_values,
        field2: Field | Face_values):

    if type(field1) is not type(field2):
        raise ValueError("Both fields must have same type")

    if field1.mesh is not field2.mesh:
        raise ValueError("Both fields must share the same mesh")

    if field1.dim != field2.dim:
        raise ValueError("Both fields must have same dimension")

    dim = field1.dim
    mesh = field1.mesh

    order = field1.order + field2.order

    tc1 = field1.total_component
    tc2 = field2.total_component

    face_operation = (
        isinstance(field1, Face_values)
        and isinstance(field2, Face_values)
    )

    result = type(field1)(mesh, order)

    for i1 in range(tc1):
        for i2 in range(tc2):

            result_component = i1 * tc2 + i2

            if face_operation:

                for d in range(dim):

                    result[result_component][d] = (
                        field1[i1][d] * field2[i2][d]
                    )

            else:

                if isinstance(field1._face_value, Face_values) & isinstance(field2._face_value, Face_values):

                    for d in range(dim):

                        result._face_value[result_component][d] = (                     # pyright: ignore
                            field1._face_value[i1][d] * field2._face_value[i2][d]       # pyright: ignore
                        )

                result[result_component] = (
                    field1[i1] * field2[i2]
                )

    return result
