"""Convert pyproj CRSs to iris coordinate systems."""

import cartopy.crs as ccrs
from iris.coord_systems import CoordSystem, GeogCS, RotatedGeogCS
from pyproj import CRS

from autogrid.crs.crs import GEOGRAPHIC_MAPPINGS, cf_rotation, ellipsoid_axes, is_wgs84
from autogrid.crs.cartopy import projection


class CartopyCS(CoordSystem):
    """An iris coordinate system for any projected CRS, delegating to cartopy.

    Iris only needs a coordinate system to provide cartopy CRSs, so this works
    for any CRS. Iris cannot save it to netCDF.
    """

    def __init__(self, crs: CRS) -> None:
        self.crs = crs

    def as_cartopy_crs(self) -> ccrs.CRS:
        return ccrs.CRS(self.crs)

    def as_cartopy_projection(self) -> ccrs.Projection:
        return projection(self.crs)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.crs.to_string()!r})"

    # A property, not an attribute, so the CRS is only converted when needed
    @property
    def grid_mapping_name(self) -> str:  # ty: ignore[invalid-property-type-override]  # pyright: ignore[reportIncompatibleVariableOverride]
        return self.crs.to_cf().get("grid_mapping_name", "unknown")


def coord_system(crs: CRS) -> CoordSystem:
    """Return the iris coordinate system for a supported pyproj CRS.

    Latitude-longitude and rotated pole CRSs must be an iris GeogCS or
    RotatedGeogCS for iris to wrap longitudes and regrid in degrees. Projected
    CRSs are wrapped.
    """
    cf = crs.to_cf()
    mapping = cf.get("grid_mapping_name")
    if mapping not in GEOGRAPHIC_MAPPINGS:
        return CartopyCS(crs)
    if is_wgs84(crs):
        geog_cs = GeogCS.from_datum("WGS84")
    else:
        geog_cs = GeogCS(*ellipsoid_axes(crs))
    if mapping == "latitude_longitude":
        return geog_cs
    return RotatedGeogCS(**cf_rotation(cf), ellipsoid=geog_cs)
