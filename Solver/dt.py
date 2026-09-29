from typing import Callable

from Fields.Fields import Field

from Schemes.BiCGStab import BiCGSTAB

from Export.export import export


def implicite(
    equation: Callable,
    x: Field,
    time: float,
    time_steps: float = 1
):
    """
    Implicit Euler time integration.

    The equation is assumed to be

        ddt(phi) = equation(phi)

    For implicit Euler:

        (phi_new - phi_old) / dt = equation(phi_new)

    which gives:

        phi_new - dt * equation(phi_new) = phi_old

    If the spatial operator contains fixed-value boundary
    contributions, the operator is separated into:

        equation(phi) = equation_h(phi) + equation(0)

    where equation(0) contains the boundary contribution.

    The equation solved by CG therefore becomes:

        A(phi_new) = B

    with

        A(phi) = phi - dt * equation_h(phi)

    and

        B = phi_old + dt * equation(0)
    """

    # --------------------------------------------------------------
    # Time step
    # --------------------------------------------------------------

    dt = time_steps

    # --------------------------------------------------------------
    # Initial field
    # --------------------------------------------------------------

    phi = x.copy()

    # --------------------------------------------------------------
    # Time
    # --------------------------------------------------------------

    t = 0.0
    i = 0

    # --------------------------------------------------------------
    # Time integration
    # --------------------------------------------------------------

    while t < time:

        # ----------------------------------------------------------
        # Export current solution
        # ----------------------------------------------------------

        filename = "t_" + str(t).replace(".", "")
        filename = "t_" + str(i)

        export(
            [phi],
            filename,
            "Result"
        )

        print(f"\nCalculating solution at t = {t + dt}")

        # ----------------------------------------------------------
        # Calculate boundary contribution.
        #
        # Create a zero field but keep the physical boundary
        # conditions.
        #
        # equation(zero) therefore gives only the contribution
        # coming from the boundary conditions.
        # ----------------------------------------------------------

        # zero = phi.copy(val_copy=False)

        # zero._values[:] = 0.0

        # if phi._Boundary_Condition is not None:

        #     zero._Boundary_Condition = (
        #         phi._Boundary_Condition.copy()
        #     )

        # bc_term = equation(zero)

        # ----------------------------------------------------------
        # Define the homogeneous linear operator.
        #
        # equation(x)
        #
        #     = homogeneous_part(x) + boundary_part
        #
        # therefore:
        #
        # homogeneous_part(x)
        #     = equation(x) - equation(0)
        # ----------------------------------------------------------

        def A(field: Field):

            return (
                field
                - dt * (
                    equation(field)
                    # - bc_term
                )
            )

        # ----------------------------------------------------------
        # Right-hand side
        #
        # phi_new - dt * equation_h(phi_new)
        #
        #     = phi_old + dt * bc_term
        # ----------------------------------------------------------

        B = phi #+ dt * bc_term

        # ----------------------------------------------------------
        # Solve the implicit system
        # ----------------------------------------------------------

        phi_new = BiCGSTAB(
            A,
            phi.copy(),
            B
        )

        # ----------------------------------------------------------
        # Update field
        # ----------------------------------------------------------

        phi = phi_new

        # ----------------------------------------------------------
        # Advance time
        # ----------------------------------------------------------

        t += dt
        i += 1

        print(f"{t} s has done")

    # --------------------------------------------------------------
    # Export final field
    # --------------------------------------------------------------

    # filename = "t_" + str(t).replace(".","")
    filename = "t_" + str(i)

    export(
        [phi],
        filename,
        "Result"
    )

    return phi
