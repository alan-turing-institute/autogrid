"""Convert pyproj CRSs to cartopy projections."""

import cartopy.crs as ccrs
from pyproj import CRS

from autogrid.crs.crs import ellipsoid_axes, is_wgs84


def globe(crs: CRS) -> ccrs.Globe:
    """Return the cartopy Globe of a supported CRS: WGS 84, or its bare ellipsoid."""
    if is_wgs84(crs):
        return ccrs.Globe(datum="WGS84")
    a, b = ellipsoid_axes(crs)
    return ccrs.Globe(semimajor_axis=a, semiminor_axis=b)


def projection(crs: CRS) -> ccrs.Projection:
    """Return the cartopy projection of a supported CRS, e.g. for plotting map axes.

    Latitude-longitude, Lambert azimuthal equal area (as used by EASE-Grid 2.0)
    and rotated pole CRSs use cartopy's own classes, which give sensible plot
    limits. Anything else is looked up by EPSG code, or wrapped if it has none.
    """
    cf = crs.to_cf()
    match cf.get("grid_mapping_name"):
        case "latitude_longitude":
            return ccrs.PlateCarree(globe=globe(crs))
        case "lambert_azimuthal_equal_area":
            return ccrs.LambertAzimuthalEqualArea(
                central_longitude=cf["longitude_of_projection_origin"],
                central_latitude=cf["latitude_of_projection_origin"],
                false_easting=cf["false_easting"],
                false_northing=cf["false_northing"],
                globe=globe(crs),
            )
        case "rotated_latitude_longitude":
            return ccrs.RotatedPole(
                pole_longitude=cf["grid_north_pole_longitude"],
                pole_latitude=cf["grid_north_pole_latitude"],
                central_rotated_longitude=cf.get("north_pole_grid_longitude", 0.0),
                globe=globe(crs),
            )
    if (epsg := crs.to_epsg()) is not None:
        return ccrs.epsg(epsg)
    return ccrs.Projection(crs)
