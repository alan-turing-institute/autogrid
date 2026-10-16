"""Convert pyproj CRSs to iris coordinate systems and units."""

import math

import cartopy.crs as ccrs
from cf_units import Unit
from iris.coord_systems import CoordSystem, GeogCS, RotatedGeogCS
from pyproj import CRS

from autogrid.crs import (
    base_geographic_crs,
    cf_rotation,
    ellipsoid_datum,
    in_degrees,
    metres_per_unit,
    prime_meridian_degrees,
)
from autogrid.crs.cartopy import projection


class CartopyMixin:
    """Delegate an iris coordinate system to cartopy, keeping the full CRS.

    Iris only needs a coordinate system to provide cartopy CRSs, so this works
    for any CRS, respecting its prime meridian and datum.
    """

    crs: CRS

    def as_cartopy_crs(self) -> ccrs.CRS:
        return ccrs.CRS(self.crs)

    def as_cartopy_projection(self) -> ccrs.Projection:
        return projection(self.crs)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.crs.to_string()!r})"


class CartopyCS(CartopyMixin, CoordSystem):
    """An iris coordinate system for any CRS. Iris cannot save it to netCDF."""

    def __init__(self, crs: CRS) -> None:
        self.crs = crs

    # A property, not an attribute, so the CRS is only converted when needed
    @property
    def grid_mapping_name(self) -> str:  # ty: ignore[invalid-property-type-override]  # pyright: ignore[reportIncompatibleVariableOverride]
        return self.crs.to_cf().get("grid_mapping_name", "unknown")


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
            **cf_rotation(crs.to_cf()),
            ellipsoid=CartopyGeogCS(base_geographic_crs(crs)),
        )
        self.crs = crs


def coord_system(crs: CRS) -> CoordSystem:
    """Return the iris coordinate system for a pyproj CRS.

    Latitude-longitude and rotated pole CRSs must be an iris GeogCS or
    RotatedGeogCS for iris to wrap longitudes and regrid in degrees, and a plain
    one if an ellipsoid, or a datum PROJ.4 can name (e.g. WGS 84, NAD83 or
    NAD27), fully describes them. Anything else is wrapped.

    Raises:
        ValueError: If a latitude-longitude or rotated pole CRS is not in
            degrees, which iris requires.
    """
    cf = crs.to_cf()
    mapping = cf.get("grid_mapping_name")
    if mapping not in ("latitude_longitude", "rotated_latitude_longitude"):
        return CartopyCS(crs)
    if not in_degrees(crs):
        msg = (
            f"Latitude-longitude CRS {crs.name!r} must be in degrees, but has "
            f"units {[axis.unit_name for axis in crs.axis_info]}."
        )
        raise ValueError(msg)
    described = ellipsoid_datum(crs)
    if described is None:
        if mapping == "latitude_longitude":
            return CartopyGeogCS(crs)
        return CartopyRotatedGeogCS(crs)
    # Keep a named datum, e.g. WGS 84 or NAD83, so iris shifts between it and others
    if described.datum is not None:
        geog_cs = GeogCS.from_datum(described.datum)
    else:
        geog_cs = GeogCS(described.semi_major_axis, described.semi_minor_axis)
    if mapping == "latitude_longitude":
        return geog_cs
    return RotatedGeogCS(**cf_rotation(cf), ellipsoid=geog_cs)


def projected_units(crs: CRS) -> str:
    """Return the units of a projected CRS's axes, as iris understands them.

    Uses the CRS's own unit name where iris recognises it, e.g. "metre" or
    "US_survey_foot", and otherwise its length in metres, e.g. "0.30479841 m".

    Raises:
        ValueError: If the axes have different units.
    """
    factor = metres_per_unit(crs)
    name = crs.axis_info[0].unit_name
    for candidate in (name, name.replace(" ", "_")):
        try:
            unit = Unit(candidate)
        except ValueError:
            continue
        if unit.is_convertible("m") and math.isclose(unit.convert(1, "m"), factor):
            return candidate
    return f"{factor!r} m"
