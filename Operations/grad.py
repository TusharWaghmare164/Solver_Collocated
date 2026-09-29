from Fields.Fields import Field
from Fields.Face_field import Face_values
from Operations import interpolation as I


def grad(field: Field | Face_values):

    '''
    Calculating Gradient
    from face values to cell center values
    '''

    if isinstance(field, Field):
        face = I.to_face(field)

    else:
        face = field

    dim = field.dim
    mesh = field.mesh
    order = field.order

    cell_neigh = mesh.cell_faces

    result = Field(mesh, order+1, initial_value=0)

    for c in range(field.total_component):

        for d in range(dim):

            result_component = (c*dim)+d

            minus = cell_neigh[:,d,0]
            plus = cell_neigh[:,d,1]

            result[result_component] = (face[c][d][plus] - face[c][d][minus]) / mesh.distances[d]

    return result
