from Fields.Fields import Field
from Fields.Face_field import Face_values
from Operations import interpolation as I


def lap(field: Field, field2: Field | Face_values | None = None):

    '''
    Calculating Laplacian
    from normal face values and to cell center values
    '''

    face = I.to_face(field)

    dim = field.dim
    mesh = field.mesh
    order = field.order

    cell_neigh = mesh.cell_faces
    face_area = mesh.face_area

    result = Field(mesh, order, initial_value=0)

    if field2 is None:

        for c in range(field.total_component):

            val = 0

            for d in range(dim):

                minus = cell_neigh[:,d,0]
                plus = cell_neigh[:,d,1]

                minus_term = 2 * (field[c] - face[c][d][minus])/ mesh.distances[d]
                plus_term = 2 * (face[c][d][plus] - field[c])/ mesh.distances[d]

                val += (plus_term * face_area[d][plus]) - (minus_term * face_area[d][minus])

            result[c] = val/mesh.cell_volume

        return result

    else:

        if not isinstance(field2, Face_values):
            face2 = I.to_face(field2)

        else:
            face2 = field2

        for c in range(field.total_component):

            val = 0

            for d in range(dim):

                minus = cell_neigh[:,d,0]
                plus = cell_neigh[:,d,1]

                minus_term = 2 * face2[c][d][minus] * (field[c] - face[c][d][minus])/ mesh.distances[d]
                plus_term = 2 * face2[c][d][plus] * (face[c][d][plus] - field[c])/ mesh.distances[d]

                val += (plus_term * face_area[d][plus]) - (minus_term * face_area[d][minus])

            result[c] = val

        return result
