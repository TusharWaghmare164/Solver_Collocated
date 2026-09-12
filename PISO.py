from Fields.Fields import Field

from Operations.gradient import grad
from Operations.divergence import div
from Operations.laplacian import lap

from Schemes.BiCGStab import BiCGSTAB

def PISO(
        U_star: Field,
        p: Field,
        rho: int|float|Field,
        dt: float,
        corrector_steps: int = 2,
):

    p = BiCGSTAB(
        A = lambda x: lap(x),
        x0 = p,
        B0 = (rho/dt)*div(U_star)
    )

    U_star._face_value -= (dt/rho._face_value)*grad(p, to_face=True)

    U_star.to_cell
