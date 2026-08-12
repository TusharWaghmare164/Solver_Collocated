"""
Bounadry Condition:
takes field as input make a boundaryCondition as a object stored within it
"""
from dataclasses import dataclass
from itertools import product
from copy import deepcopy
from typing import Self

from Settings import enums as E
from Fields.Fields import Field


# ------------------- Boundary Condition's data structure -------------------

@dataclass
class BCEntry:
    bc_type: E.BoundaryType
    bc_value: float = 0


# ------------------- Boundary Condition's data structure -------------------

@dataclass
class FaceBC:
    _configured = False


# ------------------- Helping Function -------------------

def fixedValue(value: float):

    return BCEntry(
        bc_type= E.BoundaryType.FIXED_VALUE,
        bc_value= value
        )

def Gradient(value: float):

    return BCEntry(
        bc_type= E.BoundaryType.GRADIENT,
        bc_value= value
        )

def zeroGradient():

    return BCEntry(
        bc_type= E.BoundaryType.GRADIENT,
        bc_value= 0
        )

# ------------------- Components maker -------------------

def make_component_labels(dim, order) -> list[str]:

    dirs = [f for f in E.Valid_Dir][:dim]

    return [
        "".join(p)
        for p in product(dirs, repeat=order)
    ]


# ------------------- Boundary Condition as a whole -------------------

class BoundaryCondition:

    def __init__(self, field: Field):

        self.dim = field.dim
        self.order = field.field_order

        self.__define_components_sides()

        self.empty_boundary_conditions()


    def __define_components_sides(self):

        if getattr(FaceBC, "_configured", False):
            return

        FaceBC.__annotations__ = {}

        self.faces = [f for f in E.Valid_Sides][:2*self.dim]

        for face in self.faces:
            FaceBC.__annotations__[face] = BCEntry | None
            setattr(FaceBC, face, None)

        FaceBC._configured = True


    def empty_boundary_conditions(self):

        if self.order == 0:

            self.__data__ = FaceBC()

        else :

            self.components = make_component_labels(self.dim, self.order)

            self.__data__ = {
                label: FaceBC()
                for label in self.components
            }


    def __getattr__(self, name):

        if self.order == 0:
            return getattr(self.__data__, name)

        if name in self.__data__:
            return self.__data__[name]  #pyright: ignore

        raise AttributeError(name)


    def __setattr__(self, name: str, value: BCEntry | None) -> None:

        if (self.order == 0) and (name in E.Valid_Sides):
            setattr(self.__data__, name, value)
            return

        object.__setattr__(self, name, value)


    def copy(self):

        result = BoundaryCondition.__new__(BoundaryCondition)

        result.dim = self.dim
        result.order = self.order

        if self.order != 0:
            result.components = self.components

        result.__data__ = deepcopy(self.__data__)

        return result


    def __repr__(self) -> str:

        def format_entry(entry: BCEntry | None) -> str:
            if entry is None:
                return "None"

            if entry.bc_type == E.BoundaryType.FIXED_VALUE:
                return f"fixedValue({entry.bc_value})"

            if entry.bc_type == E.BoundaryType.GRADIENT:
                if entry.bc_value == 0:
                    return "zeroGradient()"
                return f"Gradient({entry.bc_value})"

            return repr(entry)

        faces = [f for f in E.Valid_Sides][:2 * self.dim]

        if self.order == 0:
            lines = ["BoundaryCondition("]
            for face in faces:
                value = getattr(self.__data__, face)
                lines.append(f"  {face:<6} = {format_entry(value)}")
            lines.append(")")
            return "\\n".join(lines)

        lines = ["BoundaryCondition("]
        for comp in self.components:
            lines.append(f"  {comp}:")
            facebc = self.__data__[comp]    #pyright: ignore
            for face in faces:
                value = getattr(facebc, face)
                lines.append(f"    {face:<6} = {format_entry(value)}")
        lines.append(")")
        return "\\n".join(lines)
