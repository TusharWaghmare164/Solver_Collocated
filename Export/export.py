"""
ParaView export utilities.

Exports cell-centred Fields on a structured Cartesian mesh to
VTK XML StructuredGrid (.vts).

The mesh geometry is written as structured grid points and Field
values are written as CELL_DATA.

For non-scalar Fields, each tensor component is exported as a
separate scalar VTK array.
"""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

from Fields.Fields import Field
from Utilities.components import make_component_labels


# ================================================================
# Utility functions
# ================================================================


def _check_fields(fields: list[Field]) -> None:
    """
    Validate Fields before exporting.
    """

    if not fields:
        raise ValueError(
            "At least one Field is required."
        )

    for field in fields:

        if not isinstance(field, Field):
            raise TypeError(
                "All objects must be Field instances."
            )

    mesh = fields[0].mesh

    for field in fields[1:]:

        if field.mesh is not mesh:
            raise ValueError(
                "All Fields must belong to the same mesh."
            )


def _get_component_labels(field: Field) -> list[str]:
    """
    Return component labels for a Field.
    """

    if field.order == 0:
        return [""]

    return make_component_labels(
        field.dim,
        field.order
    )


def _get_field_arrays(field: Field):
    """
    Convert a Field into VTK scalar CELL_DATA arrays.

    Each component of a vector/tensor Field is exported as an
    independent scalar array.
    """

    field_name = (
        "Field"
        if field.name is None
        else str(field.name)
    )

    labels = _get_component_labels(field)

    arrays = []

    for component, label in enumerate(labels):

        if field.order == 0:

            name = field_name

        else:

            name = f"{field_name}_{label}"

        values = np.asarray(
            field._values[component],
            dtype=float
        )

        arrays.append(
            (name, values)
        )

    return arrays


def _get_mesh_points(mesh) -> np.ndarray:
    """
    Generate VTK StructuredGrid points.

    The Field is cell-centred, therefore a mesh with

        Nx, Ny, Nz

    cells has

        Nx + 1, Ny + 1, Nz + 1

    points.

    Missing dimensions are represented by a single point at
    coordinate zero.
    """

    dim = mesh.dim

    axes = []

    for axis in range(dim):

        start, end = mesh.ranges[axis]
        count = mesh.partitions[axis]

        axes.append(
            np.linspace(
                start,
                end,
                count + 1
            )
        )

    # VTK uses 3D coordinates even for 1D and 2D meshes.
    while len(axes) < 3:
        axes.append(
            np.array([0.0])
        )

    x = axes[0]
    y = axes[1]
    z = axes[2]

    points = np.empty(
        (
            len(x) *
            len(y) *
            len(z),
            3
        ),
        dtype=float
    )

    index = 0

    for k in range(len(z)):

        for j in range(len(y)):

            for i in range(len(x)):

                points[index, 0] = x[i]
                points[index, 1] = y[j]
                points[index, 2] = z[k]

                index += 1

    return points


def _format_values(values: np.ndarray) -> str:
    """
    Convert numerical values into VTK ASCII format.
    """

    return " ".join(
        f"{float(value):.17g}"
        for value in values.ravel()
    )


# ================================================================
# Main exporter
# ================================================================


def export(
    fields: list[Field],
    filename: str | Path,
    title: str = ""
) -> Path:
    """
    Export Fields to a ParaView-compatible VTK StructuredGrid file.

    Parameters
    ----------
    fields : list[Field]
        Fields to export.

        All Fields must belong to the same mesh.

    filename : str or pathlib.Path
        Name of the output file.

        The file is written inside the project's ``result``
        directory.

        Example
        -------
        export([T], "temperature")

        creates:

            result/temperature.vts

    title : str, optional
        Title stored in the VTK FieldData section.

    Returns
    -------
    pathlib.Path
        Path of the generated VTK file.

    Notes
    -----
    Scalar Fields are exported as:

        Temperature

    Vector Fields are exported as:

        Velocity_x
        Velocity_y
        Velocity_z

    Second-order tensor Fields are exported as:

        Stress_xx
        Stress_xy
        Stress_xz
        Stress_yx
        Stress_yy
        Stress_yz
        Stress_zx
        Stress_zy
        Stress_zz

    Higher-order Fields follow the same component-label convention
    defined by ``make_component_labels``.
    """

    # ------------------------------------------------------------
    # Validate Fields
    # ------------------------------------------------------------

    _check_fields(fields)

    mesh = fields[0].mesh

    # ------------------------------------------------------------
    # Prepare output path
    # ------------------------------------------------------------

    output = Path(filename)

    # Add .vts automatically
    if output.suffix.lower() != ".vts":
        output = output.with_suffix(".vts")

    # Project root = directory containing this export.py file.
    project_root = Path(__file__).resolve().parent

    result_dir = project_root.parent / "result"

    # Keep the supplied filename/path structure, but put it
    # underneath result/.
    #
    # Example:
    #
    #   "temperature"
    #
    # becomes:
    #
    #   result/temperature.vts
    #
    # and:
    #
    #   "run1/temperature"
    #
    # becomes:
    #
    #   result/run1/temperature.vts
    #
    # Absolute paths are converted to their filename so that
    # exporting cannot accidentally escape the result directory.
    if output.is_absolute():

        output = result_dir / output.name

    else:

        output = result_dir / output

    # Create result directory and any requested subdirectories.
    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # ------------------------------------------------------------
    # Mesh dimensions
    # ------------------------------------------------------------

    dim = mesh.dim

    partitions = tuple(
        mesh.partitions
    )

    nx = partitions[0]

    if dim >= 2:
        ny = partitions[1]
    else:
        ny = 1

    if dim >= 3:
        nz = partitions[2]
    else:
        nz = 1

    # Number of points in each direction.
    npx = nx + 1

    if dim >= 2:
        npy = ny + 1
    else:
        npy = 1

    if dim >= 3:
        npz = nz + 1
    else:
        npz = 1

    # ------------------------------------------------------------
    # VTK extent
    # ------------------------------------------------------------

    extent = (
        f"0 {npx - 1} "
        f"0 {npy - 1} "
        f"0 {npz - 1}"
    )

    # ------------------------------------------------------------
    # Generate mesh points
    # ------------------------------------------------------------

    points = _get_mesh_points(mesh)

    point_data = _format_values(points)

    # ------------------------------------------------------------
    # Convert Fields into VTK arrays
    # ------------------------------------------------------------

    cell_arrays = []

    for field in fields:

        arrays = _get_field_arrays(field)

        cell_arrays.extend(
            arrays
        )

    # ------------------------------------------------------------
    # Check number of cells
    # ------------------------------------------------------------

    total_cells = (
        nx *
        ny *
        nz
    )

    for name, values in cell_arrays:

        if values.size != total_cells:

            raise ValueError(
                f"Field '{name}' contains "
                f"{values.size} values, but the mesh "
                f"contains {total_cells} cells."
            )

    # ------------------------------------------------------------
    # XML-safe strings
    # ------------------------------------------------------------

    safe_title = escape(
        str(title)
    )

    # ------------------------------------------------------------
    # Write VTK file
    # ------------------------------------------------------------

    with output.open(
        "w",
        encoding="utf-8",
        newline="\n"
    ) as file:

        # --------------------------------------------------------
        # XML header
        # --------------------------------------------------------

        file.write(
            '<?xml version="1.0"?>\n'
        )

        file.write(
            '<VTKFile '
            'type="StructuredGrid" '
            'version="1.0" '
            'byte_order="LittleEndian" '
            'header_type="UInt64">\n'
        )

        # --------------------------------------------------------
        # StructuredGrid
        # --------------------------------------------------------

        file.write(
            f'  <StructuredGrid '
            f'WholeExtent="{extent}">\n'
        )

        # --------------------------------------------------------
        # Piece
        # --------------------------------------------------------

        file.write(
            f'    <Piece '
            f'Extent="{extent}">\n'
        )

        # --------------------------------------------------------
        # FieldData
        # --------------------------------------------------------

        file.write(
            "      <FieldData>\n"
        )

        file.write(
            '        <DataArray '
            'type="String" '
            'Name="Title" '
            'NumberOfTuples="1" '
            'format="ascii">'
            f'{safe_title}'
            '</DataArray>\n'
        )

        file.write(
            '        <DataArray '
            'type="Int32" '
            'Name="Dimension" '
            'NumberOfTuples="1" '
            'format="ascii">'
            f'{dim}'
            '</DataArray>\n'
        )

        file.write(
            '        <DataArray '
            'type="Int32" '
            'Name="Partitions" '
            'NumberOfComponents="3" '
            'NumberOfTuples="1" '
            'format="ascii">'
            f'{nx} {ny} {nz}'
            '</DataArray>\n'
        )

        file.write(
            "      </FieldData>\n"
        )

        # --------------------------------------------------------
        # Points
        # --------------------------------------------------------

        file.write(
            "      <Points>\n"
        )

        file.write(
            '        <DataArray '
            'type="Float64" '
            'NumberOfComponents="3" '
            'format="ascii">\n'
        )

        file.write(
            f"          {point_data}\n"
        )

        file.write(
            "        </DataArray>\n"
        )

        file.write(
            "      </Points>\n"
        )

        # --------------------------------------------------------
        # Cell data
        # --------------------------------------------------------

        file.write(
            "      <CellData>\n"
        )

        for name, values in cell_arrays:

            safe_name = escape(
                str(name)
            )

            data = _format_values(
                values
            )

            file.write(
                '        <DataArray '
                'type="Float64" '
                f'Name="{safe_name}" '
                'NumberOfComponents="1" '
                'format="ascii">\n'
            )

            file.write(
                f"          {data}\n"
            )

            file.write(
                "        </DataArray>\n"
            )

        file.write(
            "      </CellData>\n"
        )

        # --------------------------------------------------------
        # Close Piece
        # --------------------------------------------------------

        file.write(
            "    </Piece>\n"
        )

        # --------------------------------------------------------
        # Close StructuredGrid
        # --------------------------------------------------------

        file.write(
            "  </StructuredGrid>\n"
        )

        # --------------------------------------------------------
        # Close VTKFile
        # --------------------------------------------------------

        file.write(
            "</VTKFile>\n"
        )

    return output


# ================================================================
# Direct execution
# ================================================================


if __name__ == "__main__":

    raise SystemExit(
        "This module is intended to be imported and used as:\n\n"
        "    export(fields, filename, title)\n"
    )
