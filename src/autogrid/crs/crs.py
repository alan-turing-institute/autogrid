"""Helpers for pyproj CRSs.

Grids support a small set of CRSs, checked by supported_crs, whose coordinates the
regridding code can work with directly:

- latitude-longitude or rotated pole CRSs, in degrees;
- projected CRSs, in metres;

on any datum, ellipsoid or prime meridian. Coordinates in any other CRS can be
reprojected into one of these first.

Longitudes and latitudes are always on WGS 84 (EPSG:4326), so that every Grid
shares them and pyproj applies any datum shift between two Grids' CRSs. A CRS on a
bare ellipsoid or sphere with no datum, as used by many climate models, has no
defined shift, so PROJ takes its longitudes and latitudes as WGS 84 ones unchanged.

Cartopy projections of Grids' CRSs, e.g. for plotting map axes, are in
autogrid.plot, which imports cartopy.
"""

from typing import Any

from pyproj import CRS, Transformer

# Anything pyproj.CRS.from_user_input accepts, e.g. "EPSG:6931", 6931 or a CRS
CRSLike = int | str | dict[str, Any] | CRS

GEOGRAPHIC_MAPPINGS = ("latitude_longitude", "rotated_latitude_longitude")

# The CRS of every Grid's longitudes and latitudes
LONLAT = CRS.from_epsg(4326)


def supported_crs(crs: CRSLike) -> CRS:
    """Return a CRS as a pyproj CRS, checking that Grids support it.

    Raises:
        ValueError: If the CRS is not a 2D latitude-longitude or rotated pole CRS
            in degrees, or projected CRS in metres.
    """
    crs = CRS.from_user_input(crs)
    units = [axis.unit_name for axis in crs.axis_info]
    if not (crs.is_geographic or crs.is_projected):
        reason = f"is a {crs.type_name}, not a geographic or projected CRS"
    elif (
        crs.is_geographic
        and crs.to_cf().get("grid_mapping_name") not in GEOGRAPHIC_MAPPINGS
    ):
        reason = "is not a latitude-longitude or rotated pole CRS"
    elif crs.is_geographic and units != ["degree", "degree"]:
        reason = f"must be in degrees, but has units {units}"
    elif crs.is_projected and units != ["metre", "metre"]:
        reason = f"must be in metres, but has units {units}"
    else:
        return crs
    msg = (
        f"CRS {crs.name!r} {reason}. Grids support latitude-longitude and rotated "
        "pole CRSs in degrees, and projected CRSs in metres, so reproject the "
        "coordinates into one of these first."
    )
    raise ValueError(msg)


def trf_from_lonlat(crs: CRS) -> Transformer:
    """Return a transformer from WGS 84 longitude and latitude to a CRS's x and y."""
    return Transformer.from_crs(LONLAT, crs, always_xy=True)


def trf_to_lonlat(crs: CRS) -> Transformer:
    """Return a transformer from a CRS's x and y to WGS 84 longitude and latitude."""
    return Transformer.from_crs(crs, LONLAT, always_xy=True)
