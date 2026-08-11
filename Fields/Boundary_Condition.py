"""
making a class named Boundary condition
stores boundary conditions in specific formats of the field

input : field

process :
    check whether which field it belongs and then design a dicionery accroding to it to return

output : returns the whole object

why? :
    fields of variable changes fields but boundary condition should remain same
    so instead of storing it as predefined dictionery these field class's object will treat it as object
    and if the field matches the object format only then we use boundary condition unless treat it as different way

"""

from copy import deepcopy
from dataclasses import dataclass
from typing import Self, TypeAlias, cast, overload

from Settings import enums as E


@dataclass
class BCEntry:
    bc_type: E.BoundaryType
    bc_value: float = 0


BoundaryMap: TypeAlias = dict[E.Valid_Sides, BCEntry | None]


class BoundaryCondition:
    """
    Store boundary conditions together with the field type they belong to.

    The stored boundary condition may belong to a different field type than
    the current field carrying this object.
    """

    def __init__(
            self,
            ) -> None:

        self.owner_field: E.FieldType = field

        if self.is_scalar:
            self._data = self._make_boundary_format()

        elif self.is_vector:
            self._data = {
                E.Valid_Vector_Dir.X: self._make_boundary_format(),
                E.Valid_Vector_Dir.Y: self._make_boundary_format(),
            }

        elif self.is_tensor:
            self._data = {
                E.Valid_Tensor_Dir.XX: self._make_boundary_format(),
                E.Valid_Tensor_Dir.XY: self._make_boundary_format(),
                E.Valid_Tensor_Dir.YX: self._make_boundary_format(),
                E.Valid_Tensor_Dir.YY: self._make_boundary_format(),
            }

        else:
            raise ValueError(f"Unknown field type: {field}")



    @property
    def is_scalar(self) -> bool:
        return self.owner_field is E.FieldType.SCALAR

    @property
    def is_vector(self) -> bool:
        return self.owner_field is E.FieldType.VECTOR

    @property
    def is_tensor(self) -> bool:
        return self.owner_field is E.FieldType.TENSOR



    #  format of boundary condition

    def _make_boundary_format(self) -> BoundaryMap:

        BC: BoundaryMap = {
            E.Valid_Sides.LEFT: None,
            E.Valid_Sides.RIGHT: None,
            E.Valid_Sides.TOP: None,
            E.Valid_Sides.BOTTOM: None,
        }

        return BC

    #  making the validation conditions

    def _is_valid_key(self, key: BoundaryKey) -> bool:

        if isinstance(key, (E.Valid_Sides, E.Valid_Vector_Dir, E.Valid_Tensor_Dir)):
            return key in VALID_COMPONENTS[self.owner_field]

        component, face = key

        return component in VALID_COMPONENTS[self.owner_field] and face in E.Valid_Sides



    #  getting an item from the object

    @overload
    def __getitem__(self, key: E.Valid_Sides) -> BCEntry | None: ...


    @overload
    def __getitem__(
        self,
        key: E.Valid_Vector_Dir | E.Valid_Tensor_Dir,
    ) -> ScalarBoundaryMap: ...


    @overload
    def __getitem__(
        self,
        key: tuple[E.Valid_Vector_Dir | E.Valid_Tensor_Dir, E.Valid_Sides],
    ) -> BCEntry | None: ...


    def __getitem__(
        self,
        key: BoundaryKey,
    ) -> ScalarBoundaryMap | BCEntry | None:

        if isinstance(key, E.Valid_Sides):
            scalar_data = cast(ScalarBoundaryMap, self._data)
            return scalar_data[key]

        elif isinstance(key, E.Valid_Vector_Dir):
            vector_data = cast(VectorBoundaryMap, self._data)
            return vector_data[key]

        elif isinstance(key, E.Valid_Tensor_Dir):
            tensor_data = cast(TensorBoundaryMap, self._data)
            return tensor_data[key]

        else :
            component, side = key

            if isinstance(component, E.Valid_Vector_Dir) and isinstance(side, E.Valid_Sides):
                vector_data = cast(VectorBoundaryMap, self._data)
                return vector_data[component][side]

            elif isinstance(component, E.Valid_Tensor_Dir) and isinstance(side, E.Valid_Sides):
                vector_data = cast(TensorBoundaryMap, self._data)
                return vector_data[component][side]



    #  set the values inside BoundaryCondition

    def __setitem__(
        self,
        key: E.Valid_Sides | tuple[E.Valid_Vector_Dir | E.Valid_Tensor_Dir, E.Valid_Sides],
        value: BCEntry | None,
    ) -> None:

        if not self._is_valid_key(key):
            raise KeyError(f"Invalid boundary key: {key}")

        if isinstance(key, E.Valid_Sides):
            scalar_map = cast(ScalarBoundaryMap, self._data)
            scalar_map[key] = value

        else:
            component, side = key

            if isinstance(component, E.Valid_Vector_Dir):
                vector_map = cast(VectorBoundaryMap, self._data)
                vector_map[component][side] = value

            elif isinstance(component, E.Valid_Tensor_Dir):
                tensor_map = cast(TensorBoundaryMap, self._data)
                tensor_map[component][side] = value



    #  setting the new values to _data

    def set(self, _data: ScalarBoundaryMap | VectorBoundaryMap | TensorBoundaryMap) -> None:

        self._data = _data



    #  copying the Boundary Condition

    def copy(self) -> Self:
        """Return a deep copy of this boundary condition."""

        bc = type(self)(self.owner_field)
        bc._data = deepcopy(self._data)

        return bc



    #  for outside the class contain function

    def __contains__(self, item: BoundaryKey) -> bool:
        return self._is_valid_key(item)



    #  to get the keys directly as dictionery

    def keys(
        self,
    ) -> (
        tuple[E.Valid_Sides, ...]
        | tuple[
            tuple[
                E.Valid_Vector_Dir | E.Valid_Tensor_Dir,
                E.Valid_Sides,
            ],
            ...,
        ]
    ):
        """
        Return every valid boundary key for this boundary condition.
        """

        if self.is_scalar:
            return tuple(E.Valid_Sides)

        return tuple(
            (component, face)
            for component in VALID_COMPONENTS[self.owner_field]
            for face in E.Valid_Sides
        )



    #  get the items with keys too

    def items(self) -> tuple[tuple[str | tuple[str, str], BCEntry | None], ...]:
        """
        Return all (key, boundary condition) pairs.
        """

        return tuple((key, cast(BCEntry | None, self[key])) for key in self.keys())



    #  for representation

    def __repr__(self) -> str:
        """
        Return a readable representation of the BoundaryCondition.
        """

        return (
            f"{type(self).__name__}("
            f"owner_field={self.owner_field.name}, "
            f"_data={self._data})"
        )
