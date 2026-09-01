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

        self.components_map = {}

        from Fields.Boundary_Condition import BoundaryCondition as BC
        self.Boundary_Condition : BC


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


    # to face: interpolate the field value to face values

    def to_face(self) -> list[np.ndarray]:

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

        Sides = {
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

        face_values = []

        for component in range(self.total_component):

            component_faces = []

            for axe in range(mesh.dim):

                total_faces = mesh.total_faces[axe]

                face_value = np.zeros(total_faces)

                for idx in range(total_faces):

                    neighbours = mesh.face_neighbour_cells[axe][idx]

                    # ---------------- Interior ----------------

                    if -1 not in neighbours:

                        c0 = neighbours[0]
                        c1 = neighbours[1]

                        face_value[idx] = 0.5 * (
                            self[component, c0]
                            +
                            self[component, c1]
                        )

                    # ---------------- Boundary ----------------

                    else:

                        boundary_index = np.where(
                            neighbours == -1
                        )[0][0]

                        internal_cell = neighbours[
                            1 - boundary_index
                        ]

                        side = Sides[axe][boundary_index]

                        # Get correct BC
                        if self.field_order == 0:

                            bc = getattr(
                                self.Boundary_Condition,
                                side
                            )

                        else:

                            component_name = components[component]

                            component_bc = getattr(
                                self.Boundary_Condition,
                                component_name
                            )

                            bc = getattr(
                                component_bc,
                                side
                            )

                        face_value[idx] = boundary_value(
                            BC=bc,
                            internal_value=self[
                                component,
                                internal_cell
                            ], #pyright: ignore
                            opposite=(boundary_index == 1),
                            distance=mesh.distances[axe][0] / 2
                        )

                component_faces.append(face_value)

            face_values.append(component_faces)

        return face_values


def to_cell(self, face_data : None | list[np.ndarray]= None) -> npt.ndarray:

    # This function will take changed face values and interpolate them back to the cell values.
    # Main use of function will be in case of MAC Grid eg. SIMPLE and PISO method.

    if face_data is None:
        face_data = self.to_face()

    assert face_data is not None, "face_data must be provided."

    mesh = self.mesh

    for cell in range(mesh.total_cell):

        for component in range(self.total_component):

            neighbour = mesh.cell_faces[cell]
            cell_value = 0.0

            for i in range(mesh.dim):

                cell_value += face_data[component][i][neighbour[2*i]] + face_data[component][i][neighbour[(2*i)+1]]

            self[component][cell] = cell_value/ 2 * self.dim
