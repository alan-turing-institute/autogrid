"""Convert autogrid objects to iris objects."""

from functools import cache
from typing import TYPE_CHECKING, Literal

import cartopy.crs as ccrs
import numpy as np
from iris.coord_systems import CoordSystem, GeogCS, LambertAzimuthalEqualArea
from iris.coords import DimCoord
from iris.cube import Cube
from numpy.typing import ArrayLike
from pyproj import CRS

if TYPE_CHECKING:
    from autogrid.grid import Grid


class _CartopyCS(CoordSystem):
    """An iris coordinate system that delegates to cartopy.

    Iris only needs a coordinate system to provide a cartopy CRS, so this works
    for any CRS. Iris cannot save it to netCDF, and cartopy may give it poor
    plot limits.
    """

    def __init__(self, crs: CRS) -> None:
        self.crs = crs
        self.grid_mapping_name = crs.to_cf().get("grid_mapping_name", "unknown")

    def as_cartopy_crs(self) -> ccrs.Projection:
        epsg = self.crs.to_epsg()
        return ccrs.epsg(epsg) if epsg is not None else ccrs.Projection(self.crs)

    def as_cartopy_projection(self) -> ccrs.Projection:
        return self.as_cartopy_crs()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.crs.to_string()!r})"


@cache
def coord_system(crs: CRS) -> CoordSystem:
    """Return the iris coordinate system for a pyproj CRS.

    Latitude-longitude CRSs must be an iris GeogCS for iris to wrap longitudes,
    and cartopy only gives polar Lambert azimuthal equal area projections (as
    used by EASE-Grid 2.0) sensible plot limits through iris. Anything else is
    wrapped.
    """
    cf = crs.to_cf()
    match cf.get("grid_mapping_name"):
        case "latitude_longitude":
            return GeogCS(cf["semi_major_axis"], cf["semi_minor_axis"])
        case "lambert_azimuthal_equal_area":
            return LambertAzimuthalEqualArea(
                cf["latitude_of_projection_origin"],
                cf["longitude_of_projection_origin"],
                cf["false_easting"],
                cf["false_northing"],
                ellipsoid=GeogCS(cf["semi_major_axis"], cf["semi_minor_axis"]),
            )
        case _:
            return _CartopyCS(crs)


def _dim_coord(grid: "Grid", axis: Literal["x", "y"]) -> DimCoord:
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


def to_cube(grid: "Grid", data: ArrayLike | None = None) -> Cube:
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
