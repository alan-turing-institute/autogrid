"""Helpers for pyproj CRSs."""

import math
import warnings
from typing import Any

from pyproj import CRS

# Anything pyproj.CRS.from_user_input accepts, e.g. "EPSG:6931", 6931 or a CRS
CRSLike = int | str | dict[str, Any] | CRS

WGS84 = CRS.from_epsg(4326)


def base_geographic_crs(crs: CRS) -> CRS:
    """Return the geographic CRS a CRS is ultimately based on.

    Unlike CRS.geodetic_crs, this looks through derived geographic CRSs, e.g.
    rotated pole ones, to the unrotated CRS they are based on.
    """
    base = crs.geodetic_crs
    if base is None:
        msg = f"CRS {crs.name!r} has no geodetic CRS."
        raise ValueError(msg)
    while base.source_crs is not None:
        base = base.source_crs
    return base


def latlon_crs(crs: CRS) -> CRS:
    """Return a geographic CRS giving true latitude and longitude for a CRS.

    This is its base geographic CRS if that is in degrees east of Greenwich, so
    no datum shift is applied. Otherwise, e.g. for EPSG:4807 (Paris prime
    meridian, in grads), it is WGS 84.
    """
    base = base_geographic_crs(crs)
    return base if in_degrees(base) and prime_meridian_degrees(base) == 0 else WGS84


def described_by_globe(crs: CRS) -> bool:
    """Whether a cartopy Globe built from the CRS's ellipsoid fully describes it.

    This holds when the CRS has a Greenwich prime meridian, and a datum that is
    either WGS 84 or nothing more than its ellipsoid. Otherwise, e.g. for
    EPSG:4807 (Paris prime meridian) or EPSG:4277 (OSGB 1936 datum), the
    ellipsoid alone would silently drop the prime meridian or datum shift.
    """
    if crs.geodetic_crs is None:
        return False
    geodetic = base_geographic_crs(crs)
    # PROJ.4 strings can only name WGS 84 or a bare ellipsoid as the datum
    lossless = geodetic.equals(
        CRS.from_dict(proj4_dict(geodetic)), ignore_axis_order=True
    )
    return lossless and prime_meridian_degrees(geodetic) == 0


def in_degrees(crs: CRS) -> bool:
    """Whether every axis of a CRS is in degrees."""
    return all(axis.unit_name == "degree" for axis in crs.axis_info)


def full_turn(crs: CRS) -> float:
    """Return one full turn of longitude in a geographic CRS's own units.

    This is 360 for a CRS in degrees, but e.g. 400 for EPSG:4807, in grads.
    """
    for axis in crs.axis_info:
        if axis.direction in ("east", "west"):
            return 2 * math.pi / float(axis.unit_conversion_factor)
    msg = f"CRS {crs.name!r} has no longitude axis."
    raise ValueError(msg)


def prime_meridian_degrees(crs: CRS) -> float:
    """Return the longitude of the CRS's prime meridian east of Greenwich, in degrees."""
    pm = crs.prime_meridian
    if pm is None:
        return 0.0
    return math.degrees(pm.longitude * float(pm.unit_conversion_factor))


def proj4_dict(crs: CRS) -> dict[str, Any]:
    """Return the PROJ.4 parameters of a CRS, without warning that they are lossy."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "You will likely lose", UserWarning)
        return crs.to_dict()
