from Mesh.structuredMesh import StructuredMesh
from Fields.Field_maker import *

from Fields.Boundary_Condition import BoundaryCondition
import Fields.Boundary_Condition as BC
import Settings.enums as E

from Operations.divergence import div
from Operations.gradient import grad
from Operations.laplacian import lap

dim = 1
mesh = StructuredMesh([5],[(0.,1.)])

ALPHA = 0.1

Temp = Scalar(mesh,100, name= "Temp", unit="J")

BCond = BoundaryCondition(Temp)

BCond[E.Valid_Sides.LEFT] = BC.fixedValue(100)
BCond[E.Valid_Sides.RIGHT] = BC.fixedValue(300)

Temp.Boundary_Condition = BCond

equation = lambda x: ALPHA * lap(Temp)

t = 10
