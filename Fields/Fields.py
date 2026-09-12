'''
This code is for making of Field

Assumption : nth order tensor field where n > 2 is not needed
'''

import numpy as np
import numpy.typing as npt
from typing import Self

from Mesh.structuredMesh import StructuredMesh
from Settings import enums as E
from Utilities.math_utis import safe_div
from Utilities.components import make_component_labels

class Field:

    def __init__(
            self,
            mesh: StructuredMesh,
            order: int,
            initial_value: float,
            name: str | None = None,
            unit: str | None = None
            ) -> None:

        self.name = name
        self.unit = unit

        self.mesh = mesh
        self.dim = mesh.dim

        self.field_order = order

        self.total_component = mesh.dim ** order

        self._values = np.full((self.total_component, self.mesh.total_cell), initial_value, dtype=float)

        self.type_assign()

        self.components_map = {}

        from Fields.Boundary_Condition import BoundaryCondition as BC
        self.Boundary_Condition : BC
        self._face_value = None


    def type_assign(self) -> None:

        if self.field_order == 0:
            self.field_type = E.FieldType.SCALAR

        elif self.field_order == 1:
            self.field_type = E.FieldType.VECTOR

        else:
            self.field_type = E.FieldType.TENSOR


    # copying

    def copy(self, val_copy=True):

        result = self.__class__(
            self.mesh,
            self.field_order,
            0.0,
            name=self.name,
            unit=self.unit
        )

        if val_copy:
            result._values = self._values.copy()

        if self._face_value is not None:
            result._face_value = self._face_value.copy()
        else:
            result._face_value = None

        if hasattr(self, "Boundary_Condition"):
            result.Boundary_Condition = self.Boundary_Condition.copy()

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

        if isinstance(other, Field):

            if self.dim != other.dim:
                raise ValueError(
                    "Multiplication requires Fields with the same dimension."
                )

            # At least one scalar field
            if self.field_order == 0 or other.field_order == 0:

                result_order = max(
                    self.field_order,
                    other.field_order
                )

                result = Field(
                    mesh=self.mesh,
                    order=result_order,
                    initial_value=0.0,
                    name=f"({self.name}*{other.name})",
                    unit=None
                )

                result._values = (
                    self._values * other._values
                )

                return result

            # Both non-scalar -> tensor contraction
            return self.dot(other)    # pyright: ignore

        elif isinstance(other, (int, float)):

            result = self.copy(val_copy=False)
            result._values = self._values * other
            return result

        return NotImplemented


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

        if self.field_order == 0 or other.field_order == 0:
            raise ValueError(
                "Dot product requires both Fields to have non-zero order."
            )

        result_order = self.field_order + other.field_order - 2

        result = Field(
            mesh=self.mesh,
            order=result_order,
            initial_value=0.0,
            name=f"({self.name}·{other.name})",
            unit=None
        )

        a = self._values.reshape(
            (self.dim,) * self.field_order + (self.mesh.total_cell,)
        )

        b = other._values.reshape(
            (self.dim,) * other.field_order + (self.mesh.total_cell,)
        )

        result_array = np.einsum(
            a,
            list(range(self.field_order)) + [self.field_order],
            b,
            [self.field_order - 1]
            + list(range(self.field_order, self.field_order + other.field_order - 1))
            + [self.field_order],
            list(range(self.field_order - 1))
            + list(range(self.field_order, self.field_order + other.field_order - 1))
            + [self.field_order]
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

        if self.field_order != other.field_order:
            raise ValueError("Fields must have the same order.")

        return float(np.sum(self._values * other._values))

    # to face: interpolate the field value to face values

    def to_face(self) -> list[np.ndarray]:
        """Interpolate cell values to face values."""

        from Fields.Boundary_Condition import BCEntry

        def boundary_value(
            BC: BCEntry,
            internal_value: float,
            opposite: bool,
            distance: float
        ) -> float:

            if BC.bc_type == E.BoundaryType.FIXED_VALUE:
                return BC.bc_value

            elif BC.bc_type == E.BoundaryType.GRADIENT:
                if opposite:
                    return internal_value - BC.bc_value * distance
                else:
                    return internal_value + BC.bc_value * distance

            raise ValueError(
                f"Unsupported boundary condition type: {BC.bc_type}"
            )

        sides = {
            0: [E.Valid_Sides.RIGHT, E.Valid_Sides.LEFT],
            1: [E.Valid_Sides.TOP, E.Valid_Sides.BOTTOM],
            2: [E.Valid_Sides.FRONT, E.Valid_Sides.BACK]
        }

        mesh = self.mesh

        components = (
            ["scalar"]
            if self.field_order == 0
            else make_component_labels(
                self.dim,
                self.field_order
            )
        )

        # face_values[component][axis][face]
        face_values = [
            [
                np.zeros(mesh.total_faces[axe], dtype=float)
                for axe in range(mesh.dim)
            ]
            for _ in range(self.total_component)
        ]

        for axe in range(mesh.dim):

            inner_faces = mesh.inner_faces[axe]
            inner_cells = mesh.inner_cells[axe]

            # Interior faces are obtained by averaging their two neighbouring cells.
            if inner_faces.size > 0:

                c0 = inner_cells[:, 0]
                c1 = inner_cells[:, 1]

                for component in range(self.total_component):
                    face_values[component][axe][inner_faces] = (
                        0.5
                        * (
                            self._values[component, c0]
                            + self._values[component, c1]
                        )
                    )

            boundary_faces = mesh.boundary_faces[axe]
            boundary_cells = mesh.boundary_cells[axe]

            # Boundary faces require the boundary condition associated with their side.
            for idx in range(boundary_faces.size):

                face = boundary_faces[idx]
                neighbours = boundary_cells[idx]

                boundary_index = np.where(
                    neighbours == -1
                )[0][0]

                internal_cell = neighbours[1 - boundary_index]

                side = sides[axe][boundary_index]

                for component in range(self.total_component):

                    if self.field_order == 0:
                        bc = self.Boundary_Condition[side]
                    else:
                        component_name = components[component]
                        bc = self.Boundary_Condition[
                            component_name,
                            side
                        ]

                    face_values[component][axe][face] = boundary_value(
                        BC=bc,
                        internal_value=self._values[
                            component,
                            internal_cell
                        ],
                        opposite=(boundary_index == 1),
                        distance=mesh.distances[axe] / 2
                    )

        self._face_value = face_values

        return face_values


    def to_cell(
        self,
        face_data: None | list[np.ndarray] = None
    ):
        """Interpolate face values back to cell values."""

        if face_data is None:
            face_data = self.to_face()

        mesh = self.mesh

        cell_values = np.zeros_like(self._values)

        # Each cell receives the average of its two faces along every spatial axis.
        for axe in range(mesh.dim):

            minus_faces = mesh.cell_faces[:, 2 * axe]
            plus_faces = mesh.cell_faces[:, 2 * axe + 1]

            for component in range(self.total_component):
                cell_values[component] += (
                    face_data[component][axe][minus_faces]
                    +
                    face_data[component][axe][plus_faces]
                )

        self._values = cell_values / (2 * self.dim)
