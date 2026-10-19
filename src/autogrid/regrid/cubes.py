"""Build iris cubes on Grids, as the Regridder uses them."""

import numpy as np
from iris.coord_systems import RotatedGeogCS
from iris.coords import DimCoord
from iris.cube import Cube
from numpy.typing import ArrayLike, NDArray

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


def padded_coords(
    grid: Grid,
) -> tuple[tuple[DimCoord, DimCoord], tuple[tuple[int, int], tuple[int, int]]]:
    """Build the iris (y, x) coordinates of a Grid with a point added at its edges.

    Each axis gains a point at each of its outer cell bounds, so that iris
    regridders which do not extrapolate still cover the half cell between the
    outermost points and the edge of the grid. Data padded to these coordinates
    with pad_edges keeps the value of each edge cell out to its bound. A circular
    x axis is not padded, as iris wraps it, and latitudes are not padded beyond
    the poles. The padded coordinates have no bounds, as the added points are not
    cells.

    Returns:
        The padded (y, x) coordinates, and the number of points added before and
        after each of them.
    """
    y, x = _dim_coords(grid)
    limit = 90.0 if grid.is_geographic else np.inf
    y_points, y_pad = _pad_axis(grid.y_points, grid.y_bounds, limit)
    if grid.is_circular:
        x_points, x_pad = grid.x_points, (0, 0)
    else:
        x_points, x_pad = _pad_axis(grid.x_points, grid.x_bounds, np.inf)
    coords = (
        y.copy(points=y_points, bounds=None),
        x.copy(points=x_points, bounds=None),
    )
    return coords, (y_pad, x_pad)


def _pad_axis(
    points: NDArray[np.float64], bounds: NDArray[np.float64], limit: float
) -> tuple[NDArray[np.float64], tuple[int, int]]:
    """Add a point at each outer bound of an axis, unless it is beyond ±limit or
    already a point, returning the points and the number added at each end."""
    first, last = np.clip([bounds[0, 0], bounds[-1, 1]], -limit, limit)
    before = [first] if first != points[0] else []
    after = [last] if last != points[-1] else []
    return np.concatenate([before, points, after]), (len(before), len(after))


def pad_edges(
    data: np.ma.MaskedArray, pad: tuple[tuple[int, int], tuple[int, int]]
) -> np.ma.MaskedArray:
    """Pad the last two dimensions of data by repeating its edge values and mask."""
    width = [(0, 0)] * (data.ndim - 2) + list(pad)
    return np.ma.MaskedArray(
        np.pad(np.ma.getdata(data), width, mode="edge"),
        mask=np.pad(np.ma.getmaskarray(data), width, mode="edge"),
    )


def to_cube(
    grid: Grid,
    data: ArrayLike | None = None,
    coords: tuple[DimCoord, ...] | None = None,
) -> Cube:
    """Build an iris cube on a Grid, with any leading dimensions of data.

    Pass the (y, x) dim_coords of an earlier cube on the same Grid, or from
    padded_coords, to share them rather than rebuilding them, in which case they
    must not be modified.
    """
    y, x = coords or _dim_coords(grid)
    shape = (len(y.points), len(x.points)) if data is None else np.shape(data)
    n_lead = len(shape) - 2
    return Cube(
        data,
        shape=None if data is not None else shape,
        dim_coords_and_dims=[(y, n_lead), (x, n_lead + 1)],
    )
