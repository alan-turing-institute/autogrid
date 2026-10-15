"""Convert autogrid objects to iris objects."""

from functools import cache
from typing import Literal

import cartopy.crs as ccrs
import numpy as np
from iris.coord_systems import CoordSystem, GeogCS
from iris.coords import DimCoord
from iris.cube import Cube
from numpy.typing import ArrayLike
from pyproj import CRS

from autogrid.adaptors.cartopy import cartopy_projection
from autogrid.grid import Grid


class _CartopyCS(CoordSystem):
    """An iris coordinate system that delegates to cartopy.

    Iris only needs a coordinate system to provide a cartopy CRS, so this works
    for any CRS. Iris cannot save it to netCDF.
    """

    def __init__(self, crs: CRS) -> None:
        self.crs = crs
        self.grid_mapping_name = crs.to_cf().get("grid_mapping_name", "unknown")  # ty: ignore[invalid-attribute-access]

    def as_cartopy_crs(self) -> ccrs.Projection:
        return cartopy_projection(self.crs)

    def as_cartopy_projection(self) -> ccrs.Projection:
        return self.as_cartopy_crs()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.crs.to_string()!r})"


@cache
def coord_system(crs: CRS) -> CoordSystem:
    """Return the iris coordinate system for a pyproj CRS.

    Latitude-longitude CRSs must be an iris GeogCS for iris to wrap longitudes.
    Anything else is wrapped.
    """
    cf = crs.to_cf()
    if cf.get("grid_mapping_name") == "latitude_longitude":
        return GeogCS(cf["semi_major_axis"], cf["semi_minor_axis"])
    return _CartopyCS(crs)


def _dim_coord(grid: Grid, axis: Literal["x", "y"]) -> DimCoord:
    """Build an iris dimension coordinate for one axis of a Grid."""
    if grid.crs.to_cf().get("grid_mapping_name") == "rotated_latitude_longitude":
        name, units = f"grid_{'longitude' if axis == 'x' else 'latitude'}", "degrees"
    elif grid.is_geographic:
        name, units = "longitude" if axis == "x" else "latitude", "degrees"
    else:
        name, units = f"projection_{axis}_coordinate", "m"
    return DimCoord(
        grid.x_points if axis == "x" else grid.y_points,
        standard_name=name,
        units=units,
        bounds=grid.x_bounds if axis == "x" else grid.y_bounds,
        coord_system=coord_system(grid.crs),
        # Mark global longitudes as circular so regridders wrap correctly
        circular=axis == "x" and grid.is_circular,
    )


def to_cube(grid: Grid, data: ArrayLike | None = None) -> Cube:
    """Build an iris cube on a Grid, with any leading dimensions of data."""
    shape = grid.shape if data is None else np.shape(data)
    n_lead = len(shape) - 2
    return Cube(
        data,
        shape=None if data is not None else shape,
        dim_coords_and_dims=[
            (_dim_coord(grid, "y"), n_lead),
            (_dim_coord(grid, "x"), n_lead + 1),
        ],
    )
