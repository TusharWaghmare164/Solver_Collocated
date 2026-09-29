from Fields.Fields import Field
from Fields.Face_field import Face_values
from Fields.Boundary_Condition import BCEntry
from Settings import enums as E
from Schemes.BiCGStab import BiCGSTAB
from Operations.lap import lap
from Operations.div import div
from Operations.grad import grad
from Operations import interpolation as I
from typing import Callable


# ====================================

# Internal Mechanisms are written here to get correct pressure and velocity correction

# ====================================

def boundary_value(
    BC: BCEntry,
    internal_value: float,
    opposite: bool,
    distance: float
) -> float:

    if BC.bc_type == E.BoundaryType.GRADIENT:
        return BC.bc_value

    elif BC.bc_type == E.BoundaryType.FIXED_VALUE:

        if opposite:
            return (internal_value - BC.bc_value) / distance
        else:
            return (BC.bc_value - internal_value) / distance

    raise ValueError(
        f"Unsupported boundary condition type: {BC.bc_type}"
    )


# ====================================
# grad(P)
# ====================================

def grad_pressure(pressure: Field):

    sides = {
    0: [E.Valid_Sides.LEFT, E.Valid_Sides.RIGHT],
    1: [E.Valid_Sides.BOTTOM, E.Valid_Sides.TOP],
    2: [E.Valid_Sides.BACK, E.Valid_Sides.FRONT]
    }

    mesh = pressure.mesh
    dim = mesh.dim

    grad_p = Face_values(mesh, 0)

    face_neighbour = mesh.face_neighbour_cells

    assert pressure.Boundary_condition is not None


    for d in range(dim):

        neighbour = face_neighbour[d]
        minus = neighbour[:, 0]
        plus = neighbour[:, 1]

        inner_face = mesh.inner_faces[d]
        inner_minus = minus[inner_face]
        inner_plus = plus[inner_face]

        boundary_face = mesh.boundary_faces[d]
        minus_boundary = boundary_face[0]
        plus_boundary = boundary_face[1]

        bc_minus = pressure.Boundary_condition[sides[d][0]]
        bc_plus = pressure.Boundary_condition[sides[d][1]]


        grad_p[0][d][inner_face] = (
            pressure[inner_plus] - pressure[inner_minus]
        ) /mesh.distances[d]

        grad_p[0][d][minus_boundary] = boundary_value(
            BC= bc_minus,
            internal_value= pressure[plus[minus_boundary]],     # pyright: ignore
            opposite= True,
            distance= mesh.distances[d] / 2
        )

        grad_p[0][d][plus_boundary] = boundary_value(
            BC= bc_plus,
            internal_value= pressure[minus[plus_boundary]],     # pyright: ignore
            opposite= False,
            distance= mesh.distances[d] / 2
        )

    return grad_p


# ====================================
# putting U from face to cell but only in directional interpolation
# ====================================

def interp_U(field: Field):

    face = field._face_value

    assert face is not None

    mesh = face.mesh

    cell_neigh = mesh.cell_faces

    for c in range(field.total_component):

        d = c   # dimention would be same as velocity's component

        minus = cell_neigh[:,d,0]
        plus = cell_neigh[:,d,1]

        field[c] = 0.5 * (face[c][d][minus] + face[c][d][plus])

    return field


# ====================================

# Main Mechanisms starts here

# ====================================


def PISO(
    continuity_eq: Callable,
    U: Field,
    P: Field,
    rho: float | Field,
    dt: float = 0.01,
    iteration: int = 3
):
    U_star = U.copy(val_copy=True)
    P_star = P.copy(val_copy=True)

    if isinstance(rho, Field):
        rho_face = rho._face_value


    # Initialize face values once
    I.to_face(U_star)

    for _ in range(iteration):

        # Solve for incremental pressure correction
        P_corr = P.copy(val_copy=False)

        rhs = div(continuity_eq(U_star, rho)) / dt

        P_corr.assign(
            BiCGSTAB(lap, P_corr, rhs)
        )

        # Accumulate pressure correction
        P_star += P_corr

        # Correct face velocity
        if isinstance(rho, Field):
            U_star._face_value -= dt * grad_pressure(P_corr)/rho_face
        else:
            U_star._face_value -= (dt/rho) * grad_pressure(P_corr)

        # Update cell-centered velocity from corrected faces
        interp_U(U_star)

    return U_star, P_star
