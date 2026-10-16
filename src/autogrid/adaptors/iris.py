"""Convert autogrid objects to iris objects."""

import math
from typing import Literal

import cartopy.crs as ccrs
import numpy as np
from cf_units import Unit
from iris.coord_systems import CoordSystem, GeogCS, RotatedGeogCS
from iris.coords import DimCoord
from iris.cube import Cube
from numpy.typing import ArrayLike
from pyproj import CRS

from autogrid.adaptors.cartopy import cartopy_projection
from autogrid.crs import (
    base_geographic_crs,
    described_by_globe,
    prime_meridian_degrees,
    proj4_dict,
)
from autogrid.grids.grid import Grid


class CartopyMixin:
    """Delegate an iris coordinate system to cartopy, keeping the full CRS.

    Iris only needs a coordinate system to provide cartopy CRSs, so this works
    for any CRS, respecting its prime meridian and datum.
    """

    crs: CRS

    def as_cartopy_crs(self) -> ccrs.CRS:
        return ccrs.CRS(self.crs)

    def as_cartopy_projection(self) -> ccrs.Projection:
        return cartopy_projection(self.crs)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.crs.to_string()!r})"


class CartopyCS(CartopyMixin, CoordSystem):
    """An iris coordinate system for any CRS. Iris cannot save it to netCDF."""

    def __init__(self, crs: CRS) -> None:
        self.crs = crs
        self.grid_mapping_name = crs.to_cf().get("grid_mapping_name", "unknown")  # ty: ignore[invalid-attribute-access]  # pyright: ignore[reportAttributeAccessIssue]


class CartopyGeogCS(CartopyMixin, GeogCS):
    """An iris GeogCS for latitude-longitude CRSs that an ellipsoid can't describe.

    A plain GeogCS transforms with only its ellipsoid, which would drop the
    prime meridian or a datum shift. Being a GeogCS lets iris wrap longitudes.
    """

    def __init__(self, crs: CRS) -> None:
        cf = crs.to_cf()
        super().__init__(
            cf["semi_major_axis"],
            cf["semi_minor_axis"],
            longitude_of_prime_meridian=prime_meridian_degrees(crs),
        )
        self.crs = crs


class CartopyRotatedGeogCS(CartopyMixin, RotatedGeogCS):
    """An iris RotatedGeogCS for rotated pole CRSs that an ellipsoid can't describe.

    As for CartopyGeogCS, this keeps the prime meridian and datum shift that a
    plain RotatedGeogCS would drop.
    """

    def __init__(self, crs: CRS) -> None:
        super().__init__(
            **_rotation(crs), ellipsoid=CartopyGeogCS(base_geographic_crs(crs))
        )
        self.crs = crs


def _dim_coord(grid: Grid, axis: Literal["x", "y"]) -> DimCoord:
    """Build an iris dimension coordinate for one axis of a Grid."""
    if grid.crs.to_cf().get("grid_mapping_name") == "rotated_latitude_longitude":
        name, units = f"grid_{'longitude' if axis == 'x' else 'latitude'}", "degrees"
    elif grid.is_geographic:
        name, units = "longitude" if axis == "x" else "latitude", "degrees"
    else:
        name = f"projection_{axis}_coordinate"
        units = _projected_units(grid.crs)
    return DimCoord(
        grid.x_points if axis == "x" else grid.y_points,
        standard_name=name,
        units=units,
        bounds=grid.x_bounds if axis == "x" else grid.y_bounds,
        coord_system=coord_system(grid.crs),
        # Mark global longitudes as circular so regridders wrap correctly
        circular=axis == "x" and grid.is_circular,
    )


def _projected_units(crs: CRS) -> str:
    """Return the units of a projected CRS's axes, as iris understands them.

    Uses the CRS's own unit name where iris recognises it, e.g. "metre" or
    "US_survey_foot", and otherwise its length in metres, e.g. "0.30479841 m".

    Raises:
        ValueError: If the axes have different units.
    """
    factors = {float(axis.unit_conversion_factor) for axis in crs.axis_info}
    if len(factors) != 1:
        msg = (
            f"Projected CRS {crs.name!r} must have the same units on each axis, "
            f"but has units {[axis.unit_name for axis in crs.axis_info]}."
        )
        raise ValueError(msg)
    (factor,) = factors
    name = crs.axis_info[0].unit_name
    for candidate in (name, name.replace(" ", "_")):
        try:
            unit = Unit(candidate)
        except ValueError:
            continue
        if unit.is_convertible("m") and math.isclose(unit.convert(1, "m"), factor):
            return candidate
    return f"{factor!r} m"


def _rotation(crs: CRS) -> dict[str, float]:
    """Return the pole rotation of a rotated pole CRS, as RotatedGeogCS takes it."""
    cf = crs.to_cf()
    return {
        "grid_north_pole_latitude": cf["grid_north_pole_latitude"],
        "grid_north_pole_longitude": cf["grid_north_pole_longitude"],
        "north_pole_grid_longitude": cf.get("north_pole_grid_longitude", 0.0),
    }


def coord_system(crs: CRS) -> CoordSystem:
    """Return the iris coordinate system for a pyproj CRS.

    Latitude-longitude and rotated pole CRSs must be an iris GeogCS or
    RotatedGeogCS for iris to wrap longitudes and regrid in degrees, and a plain
    one if an ellipsoid, or the WGS 84 datum, fully describes them. Anything
    else is wrapped.

    Raises:
        ValueError: If a latitude-longitude or rotated pole CRS is not in
            degrees, which iris requires.
    """
    cf = crs.to_cf()
    mapping = cf.get("grid_mapping_name")
    if mapping not in ("latitude_longitude", "rotated_latitude_longitude"):
        return CartopyCS(crs)
    if any(axis.unit_name != "degree" for axis in crs.axis_info):
        msg = (
            f"Latitude-longitude CRS {crs.name!r} must be in degrees, but has "
            f"units {[axis.unit_name for axis in crs.axis_info]}."
        )
        raise ValueError(msg)
    if not described_by_globe(crs):
        if mapping == "latitude_longitude":
            return CartopyGeogCS(crs)
        return CartopyRotatedGeogCS(crs)
    # Keep a WGS 84 datum, so iris shifts between it and other datums
    if (datum := proj4_dict(base_geographic_crs(crs)).get("datum")) is not None:
        geog_cs = GeogCS.from_datum(datum)
    else:
        geog_cs = GeogCS(cf["semi_major_axis"], cf["semi_minor_axis"])
    if mapping == "latitude_longitude":
        return geog_cs
    return RotatedGeogCS(**_rotation(crs), ellipsoid=geog_cs)


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
    y, x = coords or (_dim_coord(grid, "y"), _dim_coord(grid, "x"))
    return Cube(
        data,
        shape=None if data is not None else shape,
        dim_coords_and_dims=[(y, n_lead), (x, n_lead + 1)],
    )
