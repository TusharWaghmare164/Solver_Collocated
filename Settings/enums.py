from enum import StrEnum

class FieldType(StrEnum):
    SCALAR = "scalar"
    VECTOR = "vector"
    TENSOR = "tensor"

class BoundaryType(StrEnum):
    FIXED_VALUE = "fixedValue"
    ZERO_GRADIENT = "zeroGradient"
    GRADIENT = "Gradient"

class Valid_Sides(StrEnum):
    LEFT = "left"
    RIGHT = "right"
    TOP = "top"
    BOTTOM = "bottom"

class Valid_Vector_Dir(StrEnum):
    X = "x"
    Y = "y"

class Valid_Tensor_Dir(StrEnum):
    XX = "xx"
    XY = "xy"
    YX = "yx"
    YY = "yy"
