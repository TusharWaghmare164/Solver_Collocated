from typing import Self

import numpy as np
import numpy.typing as npt

from Mesh.structuredMesh import StructuredMesh
from Utilities.math_utis import safe_div

from Fields.Fields import Field

class Face_values:

    def __init__(self, mesh: StructuredMesh, order: int) -> None:

        self.mesh = mesh
        self.dim = self.mesh.dim
        self.order = order
        self.total_component = self.dim ** self.order

        self._values = [
            [
                np.zeros(self.mesh.total_faces[d])
                for d in range(self.dim)
            ]
            for _ in range(self.total_component)
        ]


    def __setitem__(self, key, value):
        self._values[key] = value


    def __getitem__(self, key):
        return self._values[key]


    def check(self, other: Self) -> None:
        if self.mesh is not other.mesh:
            raise ValueError("Face values belong to different meshes.")

        if self.order != other.order:
            raise ValueError("Face values have different orders.")


    def field_check(self, other: Field) -> None:
        if self.mesh is not other.mesh:
            raise ValueError("Face values belong to different meshes.")

        if self.order != other.order:
            raise ValueError("Face values have different orders.")



    # copy

    def copy(self):

        result = type(self)(self.mesh, self.order)
        result._values = self._values.copy()

        return result


    # add

    def __add__(self, other: Self| Field) -> Self:

        self.check(other)

        result = type(self)(self.mesh, self.order)

        for i in range(self.total_component):
            for d in range(self.dim):
                result[i][d] = self[i][d] + other[i][d]

        return result


    __radd__ = __add__


    def __iadd__(self, other: Self):

        self.check(other)

        for i in range(self.total_component):
            for d in range(self.dim):
                self[i][d] += other[i][d]

        return self


    # substract

    def __sub__(self, other: Self) -> Self:

        self.check(other)

        result = type(self)(self.mesh, self.order)

        for i in range(self.total_component):
            for d in range(self.dim):
                result[i][d] = self[i][d] - other[i][d]

        return result


    def __rsub__(self, other: Self) -> Self:

        self.check(other)

        result = type(self)(self.mesh, self.order)

        for i in range(self.total_component):
            for d in range(self.dim):
                result[i][d] = other[i][d] - self[i][d]

        return result


    def __isub__(self, other: Self):

        self.check(other)

        for i in range(self.total_component):
            for d in range(self.dim):
                self[i][d] -= other[i][d]

        return self


    # multiplication

    def __mul__(self, other: int | float | Self):

        from Operations.mul import mul

        if isinstance(other, Face_values):
            return mul(self, other)

        else:
            result = type(self)(self.mesh, self.order)

            for i in range(self.total_component):
                for d in range(self.dim):
                    result[i][d] = self[i][d] * other

            return result


    def __rmul__(self, other: int | float | Self):

        from Operations.mul import mul

        if isinstance(other, Face_values):
            return mul(other, self)

        else:
            result = type(self)(self.mesh, self.order)

            for i in range(self.total_component):
                for d in range(self.dim):
                    result[i][d] = other * self[i][d]

            return result


    # division

    def __truediv__(self, other: int | float | Self):

        result = type(self)(self.mesh, self.order)

        if isinstance(other, (int, float)):

            for i in range(self.total_component):
                for d in range(self.dim):
                    result[i][d] = self[i][d] / other

        if isinstance(other, Face_values) and (other.order == 0):

            for i in range(self.total_component):
                for d in range(self.dim):
                    result[i][d] = safe_div(self[i][d], other[0][d])

        else:
            raise TypeError("divisor should be constant or Scalar quantity")

        return result
