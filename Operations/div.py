from Fields.Fields import Field
from Fields.Face_field import Face_values
from Operations import interpolation as I


def div(field: Field | Face_values):

    '''
    Calculating Divergence
    from normal face values to cell center values

    integration [div(phi)] dV = (phi*A)_plus - (phi*A)_minus
    '''

    if isinstance(field, Field):
        face = I.to_face(field)

    else:
        face = field

    dim = field.dim
    mesh = field.mesh
    order = field.order

    cell_neigh = mesh.cell_faces

    result = Field(mesh, order-1, initial_value=0)

    for result_component in range(result.total_component):

        val = 0

        for d in range(dim):

            c = (result_component*dim) + d

            minus = cell_neigh[:,d,0]
            plus = cell_neigh[:,d,1]

            val += (face[c][d][plus] * mesh.face_area[d][plus]) - (face[c][d][minus] * mesh.face_area[d][minus])

        result[result_component] = val

    return result
