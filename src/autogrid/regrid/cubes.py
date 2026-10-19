"""Build iris cubes on Grids, as the Regridder uses them."""

import numpy as np
from iris.coord_systems import CoordSystem, RotatedGeogCS
from iris.coords import DimCoord
from iris.cube import Cube
from numpy.typing import ArrayLike
from pyproj import Transformer

from autogrid.crs import coord_system
from autogrid.grids import Grid


def _dim_coords(grid: Grid) -> tuple[DimCoord, DimCoord]:
    """Build the iris (y, x) dimension coordinates of a Grid."""
    # Build the coordinate system once for both axes, as it is slow
    cs = coord_system(grid.crs)
    if isinstance(cs, RotatedGeogCS):
        names, units = ("grid_latitude", "grid_longitude"), "degrees"
    elif grid.is_geographic:
        names, units = ("latitude", "longitude"), "degrees"
    else:
        names = ("projection_y_coordinate", "projection_x_coordinate")
        units = "m"
    y = DimCoord(
        grid.y_points,
        standard_name=names[0],
        units=units,
        bounds=grid.y_bounds,
        coord_system=cs,
    )
    x = DimCoord(
        grid.x_points,
        standard_name=names[1],
        units=units,
        bounds=grid.x_bounds,
        coord_system=cs,
        # Mark global longitudes as circular so regridders wrap correctly
        circular=grid.is_circular,
    )
    return y, x


def sample_transformer(source: CoordSystem, target: CoordSystem) -> Transformer | None:
    """Return the transformer iris regridders use to find target points in a source CRS.

    Iris transforms target points with the cartopy CRSs of the two coordinate
    systems, or not at all if the coordinate systems are equal, in which case
    this returns None.
    """
    if source == target:
        return None
    return Transformer.from_crs(
        target.as_cartopy_crs(), source.as_cartopy_crs(), always_xy=True
    )


def to_cube(
    grid: Grid,
    data: ArrayLike | None = None,
    coords: tuple[DimCoord, ...] | None = None,
) -> Cube:
    """Build an iris cube on a Grid, with any leading dimensions of data.

    Pass the (y, x) dim_coords of an earlier cube on the same Grid to share them
    rather than rebuilding them, in which case they must not be modified.
    """
    shape = grid.shape if data is None else np.shape(data)
    n_lead = len(shape) - 2
    y, x = coords or _dim_coords(grid)
    return Cube(
        data,
        shape=None if data is not None else shape,
        dim_coords_and_dims=[(y, n_lead), (x, n_lead + 1)],
    )
