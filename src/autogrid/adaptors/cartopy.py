"""Convert autogrid objects to cartopy objects."""

import cartopy.crs as ccrs
from pyproj import CRS


def cartopy_projection(crs: CRS) -> ccrs.Projection:
    """Return the cartopy projection of a CRS, e.g. for plotting map axes.

    Latitude-longitude, Lambert azimuthal equal area (as used by EASE-Grid 2.0)
    and rotated pole CRSs use cartopy's own classes, which give sensible plot
    limits. Anything else is looked up by EPSG code, or wrapped if it has none.
    """
    cf = crs.to_cf()
    globe = ccrs.Globe(
        semimajor_axis=cf.get("semi_major_axis"),
        semiminor_axis=cf.get("semi_minor_axis"),
    )
    match cf.get("grid_mapping_name"):
        case "latitude_longitude":
            return ccrs.PlateCarree(globe=globe)
        case "lambert_azimuthal_equal_area":
            return ccrs.LambertAzimuthalEqualArea(
                central_longitude=cf["longitude_of_projection_origin"],
                central_latitude=cf["latitude_of_projection_origin"],
                false_easting=cf["false_easting"],
                false_northing=cf["false_northing"],
                globe=globe,
            )
        case "rotated_latitude_longitude":
            return ccrs.RotatedPole(
                pole_longitude=cf["grid_north_pole_longitude"],
                pole_latitude=cf["grid_north_pole_latitude"],
                central_rotated_longitude=cf.get("north_pole_grid_longitude", 0.0),
                globe=globe,
            )
    epsg = crs.to_epsg()
    return ccrs.epsg(epsg) if epsg is not None else ccrs.Projection(crs)
