from __future__ import annotations
from typing import Self

from Mesh.structuredMesh import StructuredMesh

from Utilities.math_utis import safe_div

from Fields.Boundary_Condition import BoundaryCondition

from Settings import enums as E

import numpy as np



class BaseField:

    def __init__(
            self,
            mesh: StructuredMesh,
            initial_value : float = 0.0,
            name : str = "Scalar",
            symbol : str = "phi",
            field : E.FieldType = E.FieldType.SCALAR
            ):

        self.name = name
        self.symbol = symbol

        self.field = field

        self.mesh = mesh

        self.values = np.full(
            mesh.total_cell,
            initial_value,
            dtype = float
        )

        self.Boundary_Condition = BoundaryCondition(self.field)


    # Copy function

    def copy(self) -> Self:

        result = type(self)(self.mesh)

        result.values[:] = self.values

        result.name = self.name
        result.symbol = self.symbol

        result.Boundary_Condition = self.Boundary_Condition.copy()

        result.field = self.field

        return result


    #  multiply property

    def __mul__ (self, other : int | float | Self):

        result = self.copy()

        if isinstance(other, (int,float)): result.values = other * self.values

        elif isinstance(other, type(self)): result.values = other.values * self.values

        else: return NotImplemented



        return result


    __rmul__ = __mul__


    def __imul__(self, other : int | float | Self):

        if isinstance(other, (int,float)): self.values *= other

        elif isinstance(other, type(self)): self.values *= other.values

        else: return NotImplemented

        return self



    #  add property

    def __add__(self, other : int | float | Self):

        result = self.copy()

        if isinstance(other, (int,float)): result.values = other + self.values

        elif isinstance(other, type(self)): result.values = other.values + self.values

        else: return NotImplemented



        return result


    __radd__ = __add__


    def __iadd__(self, other : int | float | Self):

        if isinstance(other, (int,float)): self.values += other

        elif isinstance(other, type(self)): self.values += other.values

        else: return NotImplemented

        return self



    #  negative Self

    def __neg__(self):

        result = self.copy()

        result.values = -self.values

        return result



    #  subtract property

    def __sub__(self, other : int | float | Self):

        result = self.copy()

        if isinstance(other, (int,float)): result.values = self.values - other

        elif isinstance(other, type(self)): result.values = self.values - other.values

        else: return NotImplemented



        return result


    def __rsub__(self, other : int | float | Self):

        result = self.copy()

        if isinstance(other, (int,float)): result.values = other - self.values

        elif isinstance(other, type(self)): result.values = other.values - self.values

        else: return NotImplemented



        return result


    def __isub__(self, other : int | float | Self):

        if isinstance(other, (int,float)): self.values -= other

        elif isinstance(other, type(self)): self.values -= other.values

        else: return NotImplemented

        return self



    # divide properties

    def  __truediv__(self, other : int | float | Self):

        result = self.copy()

        if isinstance(other, (int,float)):
            eps = np.finfo(float).eps

            result.values = self.values / np.where(
                np.abs(other) < eps,
                np.where(other >= 0, eps, -eps),
                other
            )

        elif isinstance(other, type(self)): result.values = safe_div(self.values , other.values)

        else : return NotImplemented



        return result


    def __rtruediv__(self, other : int | float | Self):

        result = self.copy()

        if isinstance(other, (int,float)):
            eps = np.finfo(float).eps

            result.values = other / np.where(
                np.abs(self.values) < eps,
                np.where(self.values >= 0, eps, -eps),
                self.values
            )

        elif isinstance(other, type(self)): result.values = safe_div(other.values , self.values)

        else: return NotImplemented

        return result
