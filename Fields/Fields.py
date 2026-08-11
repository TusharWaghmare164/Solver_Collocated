'''
This code is for making of Field

Assumption : nth order tensor field where n > 2 is not needed
'''

import numpy as np
from typing import Self

from Mesh.structuredMesh import StructuredMesh
from Settings import enums as E
from Utilities.math_utis import safe_div

from Fields.Boundary_Condition import BoundaryCondition

class Field:

    def __init__(
            self,
            mesh: StructuredMesh,
            order: int,
            initial_value: float,
            name: str | None,
            unit: str | None
            ) -> None:

        self.name = name
        self.unit = unit

        self.mesh = mesh
        self.dim = mesh.dim

        self.field_order = order

        self.total_component = mesh.dim ** order

        self._values = np.full((self.total_component, self.mesh.total_cell), initial_value, dtype=float)

        self.type_assign()
        self.Boundary_Condition = BoundaryCondition(self.field_type)    # pyright: ignore

        self.components_map = {}


    def type_assign(self) -> None:

        if self.field_order == 0:
            self.field_type = E.FieldType.SCALAR

        elif self.field_order == 1:
            self.field_type = E.FieldType.VECTOR

        else:
            self.field_type = E.FieldType.TENSOR


    # copying

    def copy(self, val_copy: bool = True) -> Self:

        result = type(self)(
            mesh = self.mesh,
            order = self.field_order,
            initial_value = 0.0,
            name = self.name,
            unit = self.unit
        )

        if val_copy:
            result._values = self._values.copy()

        return result


    # representation

    def __repr__(self) -> str:
        return (
            f"{self.field_type.name.capitalize()}Field("
            f'"{self.name}", '
            f"size={self.mesh.total_cell}, "
            f"components={self.total_component})"
        )


    # ------------------- checker function -------------------


    # this checker function will work in such way it'll check that given field input is same as self

    def check(self, other: object) -> bool:
        return (
            isinstance(other, Field)
            and self.field_order == other.field_order
            and self.dim == other.dim
        )


    # ------------------- mathematical operations -------------------


    # addition

    def __add__(self, other):

        if self.check(other):

            result = self.copy(val_copy=False)
            result._values = self._values + other._values
            return result

        else:
            raise ValueError("Addition is only supported between Field instances of the same type and size.")


    __radd__ = __add__


    def __iadd__(self, other):

        if self.check(other):

            self._values += other._values
            return self

        else:
            raise ValueError("In-place addition is only supported between Field instances of the same type and size.")


    # subtraction

    def __sub__(self, other):

        if self.check(other):

            result = self.copy(val_copy=False)
            result._values = self._values - other._values
            return result

        else:
            raise ValueError("Subtraction is only supported between Field instances of the same type and size.")


    def __rsub__(self, other):

        if self.check(other):

            result = self.copy(val_copy=False)
            result._values = other._values - self._values
            return result

        else:
            raise ValueError("Subtraction is only supported between Field instances of the same type and size.")


    def __isub__(self, other):

        if self.check(other):

            self._values -= other._values
            return self

        else:
            raise ValueError("In-place subtraction is only supported between Field instances of the same type and size.")


    # multiplication

    def __mul__(self, other):

        if self.check(other):

            result = self.copy(val_copy=False)
            result._values = self._values * other._values
            return result

        elif isinstance(other, (int, float)):

            result = self.copy(val_copy=False)
            result._values = self._values * other
            return result

        else:
            raise ValueError("Multiplication is only supported between Field instances of the same type and size.")


    __rmul__ = __mul__


    # division

    def __truediv__(self, other):

        if self.check(other):

            result = self.copy(val_copy=False)
            result._values = safe_div(self._values, other._values)
            return result

        elif isinstance(other, (int, float)):

            result = self.copy(val_copy=False)
            result._values = self._values / other
            return result

        else:
            raise ValueError("Division is only supported between Field instances of the same type and size.")


    def __rtruediv__(self, other):

        if self.check(other):

            result = self.copy(val_copy=False)
            result._values = safe_div(other._values, self._values)
            return result

        elif isinstance(other, (int, float)):

            result = self.copy(val_copy=False)
            result._values = other / self._values
            return result

        else:
            raise ValueError("Division is only supported between Field instances of the same type and size.")


    # negative

    def __neg__(self):
        result = self.copy(val_copy=False)
        result._values = -self._values
        return result


    # getters and setters

    def __getitem__(self, key):
        return self._values[key]

    def __setitem__(self, key, value):
        self._values[key] = value

    @property
    def shape(self):
        return self._values.shape


    # equality

    def __eq__(self, other):
        return (
            isinstance(other, Field)
            and self.mesh is other.mesh
            and np.array_equal(self._values, other._values)
        )
