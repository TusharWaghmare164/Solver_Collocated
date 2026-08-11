"""
This Code is made to make the field defined in Fields.py

Here will be three functions, one for each type of field (scalar, vector, tensor)

also with dictionery of component naming for each type of field's each layer of Matrix (2nd axis (not row oe column) of the matrix)
"""

from itertools import product

from Fields.Fields import Field
from Mesh.structuredMesh import StructuredMesh


# ------------------- Components Assigning -------------------


# making iterable names to appoint to each components of Field's values

def make_component_labels(dim, order) -> list[str]:

    dirs = ["x", "y", "z"][:dim]

    return [
        "".join(p)
        for p in product(dirs, repeat=order)
    ]


# now by use of setattr we'll assign each layer of Field a name if order != 0

def attach_components(field: Field):

    field.components_map = {}

    if field.field_order == 0: return field

    labels = make_component_labels(field.dim, field.field_order)

    cls = field.__class__

    for row, label in enumerate(labels):

        field.components_map[label] = row

        if not hasattr(cls, label):

            setattr(
                cls,
                label,
                property(
                    lambda self, r=row: self._values[r]
                )
            )

    return field


# ------------------- Field making -------------------

def Scalar(
        mesh: StructuredMesh,
        initial_value: float,
        name: str | None,
        unit: str | None
        ) -> Field:

    return Field(mesh, 0, initial_value, name, unit)


def Vector(
        mesh: StructuredMesh,
        initial_value: float,
        name: str | None,
        unit: str | None
        ) -> Field:

    return Field(mesh, 1, initial_value, name, unit)


def Tensor(
        mesh: StructuredMesh,
        order: int,
        initial_value: float,
        name: str | None,
        unit: str | None
        ) -> Field:

    return Field(mesh, order, initial_value, name, unit)
