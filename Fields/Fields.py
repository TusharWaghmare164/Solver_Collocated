'''
This code is for making of Field

Assumption : nth order tensor field where n > 2 is not needed
'''

import numpy as np
from typing import Self

from Mesh.structuredMesh import StructuredMesh
from Settings import enums as E

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

        self.field_type : E.FieldType

        self.mesh = mesh

        self.field_order = order

        self.total_component = (order * mesh.dim) or 1

        self._values = np.full((self.total_component, self.mesh.total_cell), initial_value)

        self.type_assign()

        self.Boundary_Condition = BoundaryCondition(self.field_type)


    def type_assign(self) -> None:

        if self.field_order == 0:
            self.field_type = E.FieldType.SCALAR

        elif self.field_order == 1:
            self.field_type = E.FieldType.VECTOR

        else:
            self.field_type = E.FieldType.TENSOR


    # copying

    def copy(self) -> Self:

        result = type(self)(
            mesh = self.mesh,
            dim = self.dim,
            order = self.field_order,
            initial_value = 0.0,
            name = self.name,
            unit = self.unit
        )


    # representation

    def __repr__(self) -> str:
        return (
            f"{self.field_type.name.capitalize()}Field("
            f'"{self.name}", '
            f"size={self.mesh.total_cell}, "
            f"components={self.total_component})"
        )


    # ------------------- mathematical operations -------------------

    def __add__(self, other):

        if isinstance(other, Field) and other.total_component == self.total_component:

            result = type
