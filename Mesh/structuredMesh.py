# This file is to make structured mesh

import numpy as np
from collections.abc import Sequence

from Settings import enums as E

class StructuredMesh:

    def __init__(
            self,
            xrange : tuple[float, float],
            Nx : int,
            yrange : tuple[float, float],
            Ny : int):

        self.x0, self.x1 = xrange
        self.y0, self.y1 = yrange
        self.Nx = Nx              # no of cells in x direction
        self.Ny = Ny              # no of cells in y direction

        self.total_cell = self.Nx * self.Ny
        self.total_face_x = (self.Nx + 1) * self.Ny
        self.total_face_y = self.Nx * (self.Ny + 1)
        self.total_points = (self.Nx + 1) * (self.Ny + 1)

        self.dx = (self.x1 - self.x0)/self.Nx
        self.dy = (self.y1 - self.y0)/self.Ny

        self.Ax = 1 * self.dy
        self.Ay = 1 * self.dx

        self.Volume = self.dx * self.dy * 1


        # making the coordinates for the cell center
        self.cells = self.built(
            [self.x0 + (self.dx/2), self.x1-(self.dx/2)],
            self.Nx,
            [self.y0 + (self.dy/2), self.y1-(self.dy/2)],
            self.Ny)


        self.faces_x = self.built(
            xrange,
            self.Nx + 1,
            [self.y0 + (self.dy/2), self.y1-(self.dy/2)],
            self.Ny)


        self.faces_y = self.built(
            [self.x0 + (self.dx/2), self.x1-(self.dx/2)],
            self.Nx,
            yrange,
            self.Ny + 1)

        # making the neighbours for the cells
        self.cell_neigh = np.full((self.total_cell,4), -1, dtype = np.int32)

        # this specific array will store all face's index for each cell in E,W,N,S order for vector
        self.cell_faces = np.full((self.total_cell,4), -1, dtype = np.int32)

        # this specific array will store all vertical face's neightbours in cellE, cellW for face neighbour relations
        self.x_face_neigh = np.full((self.total_face_x,2), -1, dtype = np.int32)

        # this specific array will store all horizontal face's neightbours in cellN, cellS for face neighbour relations
        self.y_face_neigh = np.full((self.total_face_y,2), -1, dtype = np.int32)

        self.boundary_cells : dict[str,list[int]] = {
            E.Valid_Sides.LEFT:   [],
            E.Valid_Sides.RIGHT:  [],
            E.Valid_Sides.TOP:    [],
            E.Valid_Sides.BOTTOM: []
        }

        self.boundary_faces : dict[str,list[int]] = {
            E.Valid_Sides.LEFT:   [],
            E.Valid_Sides.RIGHT:  [],
            E.Valid_Sides.TOP:    [],
            E.Valid_Sides.BOTTOM: []
        }

        self.neigh()



    #  assigning the coordinates

    @staticmethod
    def built(
        xlim : Sequence[float],
        Nx : int,
        ylim : Sequence[float],
        Ny : int):

        x = np.linspace(xlim[0], xlim[1], Nx)
        y = np.linspace(ylim[0], ylim[1], Ny)

        X,Y = np.meshgrid(x,y)

        nodes = np.column_stack((X.ravel(),Y.ravel()))

        return nodes



    #  neighbouring data all forms

    def neigh(self):

        for idx in range(self.total_cell):

            i = idx % self.Nx       # index on x
            j = idx // self.Nx      # index on y

            west = (self.Nx + 1) * j + i
            east = west + 1

            south = (self.Nx) * j + i
            north = south + self.Nx

            self.cell_faces[idx,:] = [east, west, north, south]

            # first east

            if i < self.Nx - 1:
                self.cell_neigh[idx,0] = idx + 1
                self.x_face_neigh[east,1] = idx
            else:
                self.boundary_cells[E.Valid_Sides.RIGHT].append(idx)
                self.boundary_faces[E.Valid_Sides.RIGHT].append(east)

            # second west

            if i > 0:
                self.cell_neigh[idx,1] = idx - 1
                self.x_face_neigh[west,0] = idx
            else:
                self.boundary_cells[E.Valid_Sides.LEFT].append(idx)
                self.boundary_faces[E.Valid_Sides.LEFT].append(west)

            #  third north

            if j < self.Ny - 1:
                self.cell_neigh[idx,2] = idx + self.Nx
                self.y_face_neigh[north,2] = idx
            else:
                self.boundary_cells[E.Valid_Sides.TOP].append(idx)
                self.boundary_faces[E.Valid_Sides.TOP].append(north)

            # fourth south

            if j > 0:
                self.cell_neigh[idx,3] = idx - self.Nx
                self.y_face_neigh[south,3] = idx
            else:
                self.boundary_cells[E.Valid_Sides.BOTTOM].append(idx)
                self.boundary_faces[E.Valid_Sides.BOTTOM].append(south)


        return
