"""
Structured Cartesian Mesh.

Provides:
        - cell center coordinates
        - face coordinates
        - face neighbour cell index
        - Volume of cell
        - face area list

"""

# library
import numpy as np
from collections.abc import Sequence

# modules
from Settings import enums as E


# ========================== Class StructuredMesh ==========================

class StructuredMesh:

    """
    StructuredMesh

    inputs:
        - partitions : which stores the number of cells in each direction, e.g. [Nx, Ny, ...]
        - ranges : which stores the range of each direction, e.g. [(x0, x1), (y0, y1), ...]

    outputs:
        - cell center (cells) and face coordinates (faces)
        - face neighbour cells list
        - Volume of cell
        - face area list
        - dimension of mesh
    """

    def __init__(
            self,
            partitions : Sequence[int] | None = None,
            ranges : Sequence[tuple[float, float]] | None = None,
            ) -> None:

        self.partitions, self.ranges = self._set_var(partitions, ranges)

        assert self.partitions is not None and self.ranges is not None, "partitions and ranges must be provided."

        self._set_counts(self.partitions)

        self._set_properties(self.ranges, self.partitions)

        self._set_coords(self.ranges, self.partitions)

        self._cell_faces()

        self._neighbour_cells()


    # ========================== Set Variables ==========================

    def _set_var(self, partitions, ranges) -> tuple[Sequence[int], Sequence[tuple[float, float]]]:

        if partitions is None and ranges is None:
            partitions = (10, 10)
            ranges = [(0, 1), (0, 1)]

        elif partitions is None:
            partitions = (10,) * len(ranges)    # pyright: ignore

        elif ranges is None:
            ranges = [(0, 1)] * len(partitions)

        if len(partitions) != len(ranges):      # pyright: ignore
            raise ValueError(
                "partitions and ranges must have the same number of dimensions."
            )

        if len(partitions) > 3 or len(ranges) > 3:      # pyright: ignore
            raise ValueError(
                "StructuredMesh only supports up to 3 dimensions."
            )

        return partitions, ranges


    # ========================== Set Counts, Properties and Coordinates ==========================

    def _set_counts(self, partitions):
        self.partitions = np.array(partitions, dtype=int)

        self.dim = len(partitions)
        self.total_cell = np.prod(self.partitions)

        self.total_faces = [
            np.prod([
                p + 1 if i == j else p
                for j, p in enumerate(self.partitions)
            ])
            for i in range(self.dim)
        ]


    # ========================== Set Properties ==========================

    def _set_properties(self, ranges, partitions):

         self.distances = [
             (r[1] - r[0]) / p
             for r, p in zip(ranges, partitions)
         ]

         self.cell_volume = np.full(self.total_cell, np.prod(self.distances))

         self.face_area = [
             np.full(
                 self.total_faces[i],
                 np.prod([d for j, d in enumerate(self.distances) if j != i]) if self.dim > 1 else 1.0,
             )
             for i in range(self.dim)
         ]


    # ========================== Set Coordinates ==========================

    def _set_coords(self, ranges, partitions):
        # for cells (cell center coordinates) it'll start from offset of half distance and end at offset of half distance from the end

            self.cells = self.built_coords(
                tuple(
                    (r[0] + d / 2, r[1] - d / 2)
                    for r, d in zip(ranges, self.distances)
                ),
                partitions,
            )

            # in such way bulding face coordintes
            # if faces[0] then it'll be in partition[0] + 1 and only ranges[0] will be as it is but other ranges will be offset by half distance and end at offset of half distance from the end

            face_ranges = [
                [
                    (r[0], r[1]) if i == j
                    else (r[0] + d / 2, r[1] - d / 2)
                    for j, (r, d) in enumerate(zip(ranges, self.distances))
                ]
                for i in range(self.dim)
            ]

            face_partitions = [
                [
                    (p + 1) if i == j else p
                    for j, p in enumerate(partitions)
                ]
                for i in range(self.dim)
            ]

            self.faces = [
                self.built_coords(r, p)
                for r, p in zip(face_ranges, face_partitions)
            ]


    @staticmethod
    def built_coords(ranges, partitions):
        axes = [
            np.linspace(r[0], r[1], p)
            for r, p in zip(ranges, partitions)
        ]

        grids = np.meshgrid(*axes, indexing="ij")
        return np.stack(grids, axis=-1).reshape(-1, len(axes))


    # ========================== Set Cell Faces ==========================

    def _cell_faces(self):
        """
        For every cell, return the indices of its faces.

        1D : [W, E]
        2D : [W, E, S, N]
        3D : [W, E, S, N, B, F]
        """

        p = np.array(self.partitions)
        cell_shape = tuple(p)

        # coordinates of every cell
        cell_coords = np.array(
            np.unravel_index(np.arange(self.total_cell), cell_shape)
        ).T                                    # (ncell, dim)

        faces = []

        for axis in range(self.dim):

            face_shape = p.copy()
            face_shape[axis] += 1

            # minus face (W/S/B)
            minus = np.ravel_multi_index(
                cell_coords.T,
                tuple(face_shape)
            )

            # plus face (E/N/F)
            plus_coords = cell_coords.copy()
            plus_coords[:, axis] += 1

            plus = np.ravel_multi_index(
                plus_coords.T,
                tuple(face_shape)
            )

            faces.extend([minus, plus])

        self.cell_faces = np.column_stack(faces)


    # ========================== Set Neighbour Cells ==========================

    def _neighbour_cells(self):

        p = np.array(self.partitions)
        cell_shape = tuple(p)

        self.face_neighbour_cells = []

        self.inner_faces = []
        self.inner_cells = []

        self.boundary_faces = []
        self.boundary_cells = []

        for axis in range(self.dim):

            face_shape = p.copy()
            face_shape[axis] += 1
            face_shape = tuple(face_shape)

            nfaces = np.prod(face_shape)

            coords = np.array(
                np.unravel_index(
                    np.arange(nfaces),
                    face_shape
                )
            ).T

            plus = np.full(nfaces, -1, dtype=int)
            minus = np.full(nfaces, -1, dtype=int)

            mask = coords[:, axis] < p[axis]

            if np.any(mask):
                plus[mask] = np.ravel_multi_index(
                    coords[mask].T,
                    cell_shape
                )

            c = coords.copy()
            c[:, axis] -= 1

            mask = c[:, axis] >= 0

            if np.any(mask):
                minus[mask] = np.ravel_multi_index(
                    c[mask].T,
                    cell_shape
                )

            neighbours = np.column_stack((minus, plus))

            # Store all neighbours as before
            self.face_neighbour_cells.append(neighbours)

            # Identify interior and boundary faces
            inner_mask = (minus != -1) & (plus != -1)
            boundary_mask = ~inner_mask

            # Store interior face information
            self.inner_faces.append(
                np.where(inner_mask)[0]
            )

            self.inner_cells.append(
                neighbours[inner_mask]
            )

            # Store boundary face information
            self.boundary_faces.append(
                np.where(boundary_mask)[0]
            )

            self.boundary_cells.append(
                neighbours[boundary_mask]
            )
