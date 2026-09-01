"""
This code will return the interpolation matrix for given mesh
"""

import numpy as np
from scipy.sparse import coo_matrix
from enum import StrEnum

from Mesh.structuredMesh import StructuredMesh
from Fields.Fields import Field
from Settings import enums as E

#
class Sides(StrEnum):

Sides = {
    0: [E.Valid_Sides.RIGHT,E.Valid_Sides.LEFT],
    1: [E.Valid_Sides.TOP,E.Valid_Sides.BOTTOM],
    2: [E.Valid_Sides.FRONT,E.Valid_Sides.BACK]
}


def interp(field: Field):

    mesh = field.mesh
    total_cells = mesh.total_cell

    I = []

    for axe in range(mesh.dim):

        total_faces = mesh.total_faces[axe]
        face_neighbour = mesh.face_neighbour_cells[axe]

        Intp = np.zeros(total_faces)

        for idx in range(total_faces):

            if -1 not in face_neighbour[idx]:

                Intp[idx] = 0.5 * (
                    field[face_neighbour[idx][0]] +
                    field[face_neighbour[idx][1]]
                    )

            else:




        # row = []
        # column = []
        # value = []

        # b_rows = []
        # b_columns = []
        # b_values = []

        # total_faces = mesh.total_faces[axe]
        # face_neighbour = mesh.face_neighbour_cells[axe]

        # for idx in range(total_faces):

        #     if -1 not in face_neighbour[idx]:

        #         row += 2*[idx]
        #         column += face_neighbour[idx]
        #         value += 2*[1/2]

        #     else:

