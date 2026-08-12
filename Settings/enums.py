from enum import StrEnum

class FieldType(StrEnum):
    SCALAR = "scalar"
    VECTOR = "vector"
    TENSOR = "tensor"

class BoundaryType(StrEnum):
    FIXED_VALUE = "fixedValue"
    GRADIENT = "Gradient"

class Valid_Sides(StrEnum):
    LEFT = "left"
    RIGHT = "right"
    TOP = "top"
    BOTTOM = "bottom"
    FRONT = "front"
    BACK = "back"

class Valid_Dir(StrEnum):
    X = "x"
    Y = "y"
    Z = "z"
