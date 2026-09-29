'''
This code is for making of Field

Assumption : nth order tensor field where n > 2 is not needed
'''

import numpy as np
from typing import Self

from Mesh.structuredMesh import StructuredMesh
from Settings import enums as E
from Utilities.math_utis import safe_div

class Field:

    def __init__(
            self,
            mesh: StructuredMesh,
            order: int,
            initial_value: float | None = None,
            name: str | None = None,
            unit: str | None = None
            ) -> None:

        self.name = name
        self.unit = unit

        self.mesh = mesh
        self.dim = mesh.dim

        self.order = order

        self.total_component = mesh.dim ** order

        self._values = np.full((self.total_component, self.mesh.total_cell), 0 if initial_value is None else initial_value, dtype=float)

        self.type_assign()

        self.components_map = {}


        self._Boundary_Condition = None
        self._face_value = None


    @property
    def Boundary_condition(self):
        return self._Boundary_Condition


    from Fields.Boundary_Condition import BoundaryCondition as BC

    @Boundary_condition.setter
    def Boundary_condition(self, value: BC):
        self._Boundary_Condition = value

        from Operations import interpolation as I
        self._face_value = I.to_face(self)


    def type_assign(self) -> None:

        if self.order == 0:
            self.field_type = E.FieldType.SCALAR

        elif self.order == 1:
            self.field_type = E.FieldType.VECTOR

        else:
            self.field_type = E.FieldType.TENSOR


    # copying

    def copy(self, val_copy=True):

        result = self.__class__(
            self.mesh,
            self.order,
            0.0,
            name=self.name,
            unit=self.unit
        )

        result._Boundary_Condition = (
            self._Boundary_Condition.copy()
            if self._Boundary_Condition is not None
            else None
        )

        if val_copy:
            result._values = self._values.copy()
            result._face_value = (
                self._face_value.copy()
                if self._face_value is not None
                else None
            )
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
            and self.order == other.order
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

            from Operations import interpolation as I
            I.to_face(self)
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

            from Operations import interpolation as I
            I.to_face(self)
            return self

        else:
            raise ValueError("In-place subtraction is only supported between Field instances of the same type and size.")


    # multiplication

    def __mul__(self, other):

        from Operations.mul import mul
        if isinstance(other, Field):

            if self.dim != other.dim:
                raise ValueError(
                    "Multiplication requires Fields with the same dimension."
                )

            # At least one scalar field
            result = mul(self, other)

            return result

        elif isinstance(other, (int, float)):

            result = self.copy(val_copy=False)
            result._values = self._values * other
            return result

        return NotImplemented


    __rmul__ = __mul__


    # division

    def __truediv__(self, other):

        from Operations.mul import mul

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

        from Operations import interpolation as I
        I.to_face(self)
        result._values = -self._values
        return result


    # getters and setters

    def __getitem__(self, key):
        return self._values[key]

    def __setitem__(self, key, value):
        self._values[key] = value

        if self._Boundary_Condition is not None:
            from Operations import interpolation as I
            I.to_face(self)

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


    def assign(self, other: Self):
        if not self.check(other):
            raise ValueError("Incompatible fields.")

        self._values = other._values.copy()

        if other._face_value is not None:
            self._face_value = other._face_value.copy()
        else:
            self._face_value = None

        return self


    # Transpose
    @property
    def T(self):

        result = self.copy()
        result._values = self._values.T

        return result


    def __matmul__(self, other : Self):

        result = self._values @ other._values

        return result


    def norm(self):

        return np.linalg.norm(self._values)


    def dot(self, other: Self):

        if not isinstance(other, Field):
            raise TypeError(
                "Dot product is only supported between Field instances."
            )

        if self.dim != other.dim:
            raise ValueError(
                "Dot product requires Fields with the same spatial dimension."
            )

        if self.order == 0 or other.order == 0:
            raise ValueError(
                "Dot product requires both Fields to have non-zero order."
            )

        result_order = self.order + other.order - 2

        result = Field(
            mesh=self.mesh,
            order=result_order,
            initial_value=0.0,
            name=f"({self.name}·{other.name})",
            unit=None
        )

        a = self._values.reshape(
            (self.dim,) * self.order + (self.mesh.total_cell,)
        )

        b = other._values.reshape(
            (self.dim,) * other.order + (self.mesh.total_cell,)
        )

        result_array = np.einsum(
            a,
            list(range(self.order)) + [self.order],
            b,
            [self.order - 1]
            + list(range(self.order, self.order + other.order - 1))
            + [self.order],
            list(range(self.order - 1))
            + list(range(self.order, self.order + other.order - 1))
            + [self.order]
        )

        result._values = result_array.reshape(
            result.total_component,
            self.mesh.total_cell
        )

        return result


    def inner(self, other: Self) -> float:
        if not isinstance(other, Field):
            raise TypeError("inner() requires another Field.")

        if self.mesh is not other.mesh:
            raise ValueError("Fields must belong to the same mesh.")

        if self.order != other.order:
            raise ValueError("Fields must have the same order.")

        return float(np.sum(self._values * other._values))
