"""Convert pyproj CRSs to cartopy projections."""

import cartopy.crs as ccrs
from pyproj import CRS

from autogrid.crs import cf_rotation, ellipsoid_datum, full_turn, metres_per_unit


def projection(crs: CRS) -> ccrs.Projection:
    """Return the cartopy projection of a CRS, e.g. for plotting map axes.

    Latitude-longitude, Lambert azimuthal equal area (as used by EASE-Grid 2.0)
    and rotated pole CRSs use cartopy's own classes, which give sensible plot
    limits, if a cartopy Globe can describe them. PlateCarree cannot keep a datum
    other than WGS 84, so latitude-longitude CRSs with one are wrapped. Cartopy's
    Lambert azimuthal equal area class works in metres, so for other linear units
    the CRS is wrapped with the same limits converted to its own units. Anything
    else is looked up by EPSG code, or wrapped if it has none.
    """
    described = ellipsoid_datum(crs)
    if described is None:
        return _wrapped_projection(crs)
    cf = crs.to_cf()
    # Keep a named datum, e.g. NAD27, so cartopy shifts between it and others.
    # PROJ treats a datum given with explicit ellipsoid axes as unknown, so give
    # the datum alone
    if described.datum is not None:
        globe = ccrs.Globe(datum=described.datum, ellipse=None)
    else:
        globe = ccrs.Globe(
            semimajor_axis=described.semi_major_axis,
            semiminor_axis=described.semi_minor_axis,
        )
    match cf.get("grid_mapping_name"):
        # PlateCarree always sets a numeric prime meridian, with which PROJ treats
        # any datum as unknown, so other datums are wrapped below
        case "latitude_longitude" if described.datum in (None, "WGS84"):
            return ccrs.PlateCarree(globe=globe)
        case "lambert_azimuthal_equal_area":
            # False easting and northing are given in the CRS's own units
            metres = metres_per_unit(crs)
            laea = ccrs.LambertAzimuthalEqualArea(
                central_longitude=cf["longitude_of_projection_origin"],
                central_latitude=cf["latitude_of_projection_origin"],
                false_easting=cf["false_easting"] * metres,
                false_northing=cf["false_northing"] * metres,
                globe=globe,
            )
            if metres == 1.0:
                return laea
            projection = ccrs.Projection(crs)
            limits = (*laea.x_limits, *laea.y_limits)
            projection.bounds = tuple(limit / metres for limit in limits)
            projection.threshold = laea.threshold / metres
            return projection
        case "rotated_latitude_longitude":
            rotation = cf_rotation(cf)
            return ccrs.RotatedPole(
                pole_longitude=rotation["grid_north_pole_longitude"],
                pole_latitude=rotation["grid_north_pole_latitude"],
                central_rotated_longitude=rotation["north_pole_grid_longitude"],
                globe=globe,
            )
    return _wrapped_projection(crs)


def _wrapped_projection(crs: CRS) -> ccrs.Projection:
    """Return a projected CRS's cartopy projection by EPSG code, or wrap any CRS."""
    if crs.is_projected and (epsg := crs.to_epsg()) is not None:
        return ccrs.epsg(epsg)
    projection = ccrs.Projection(crs)
    if crs.is_geographic:
        # Plot the whole globe, not just the CRS's area of use, in its own units
        half_turn = full_turn(crs) / 2
        projection.bounds = (-half_turn, half_turn, -half_turn / 2, half_turn / 2)
        projection.threshold = half_turn / 100
    return projection
