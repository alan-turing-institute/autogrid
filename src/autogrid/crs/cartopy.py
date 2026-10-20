"""Convert Grids to cartopy projections."""

import math

import cartopy.crs as ccrs
from pyproj import CRS

from autogrid.grids import Grid


def globe(crs: CRS) -> ccrs.Globe:
    """Return the cartopy Globe of a CRS: WGS 84, or else its ellipsoid.

    A datum other than WGS 84 is approximated by its ellipsoid alone, ignoring its
    datum shift. This is typically no more than a few hundred metres, too little to
    see on a map.
    """
    datum = crs.datum
    if datum is not None and datum.name.startswith("World Geodetic System 1984"):
        return ccrs.Globe(datum="WGS84")
    ellipsoid = crs.ellipsoid
    if ellipsoid is None:
        msg = f"CRS {crs.name!r} has no ellipsoid."
        raise ValueError(msg)
    return ccrs.Globe(
        semimajor_axis=ellipsoid.semi_major_metre,
        semiminor_axis=ellipsoid.semi_minor_metre,
    )


def projection(grid: Grid) -> ccrs.Projection:
    """Return the cartopy projection of a Grid's CRS, e.g. for plotting map axes.

    Latitude-longitude, Lambert azimuthal equal area (as used by EASE-Grid 2.0)
    and rotated pole CRSs use cartopy's own classes, which give sensible plot
    limits. Anything else is looked up by EPSG code, or wrapped if it has none.
    """
    crs = grid.crs
    cf = crs.to_cf()
    # Correct for the prime meridian of the projection as Cartopy requires longitudes
    # from Greenwich
    meridian = 0.0
    if (pm := crs.prime_meridian) is not None:
        meridian = math.degrees(pm.longitude * float(pm.unit_conversion_factor))
    match cf.get("grid_mapping_name"):
        case "latitude_longitude":
            return ccrs.PlateCarree(central_longitude=meridian, globe=globe(crs))
        case "lambert_azimuthal_equal_area":
            return ccrs.LambertAzimuthalEqualArea(
                central_longitude=cf["longitude_of_projection_origin"] + meridian,
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
