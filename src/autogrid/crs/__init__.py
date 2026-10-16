"""Helpers for pyproj CRSs, and their cartopy and iris equivalents.

- cartopy: cartopy projections of CRSs, e.g. for plotting map axes.
- iris: iris coordinate systems of CRSs.
"""

import math
import warnings
from dataclasses import dataclass
from typing import Any

from pyproj import CRS, Transformer

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


def true_geographic_crs(crs: CRS) -> CRS:
    """Return a geographic CRS giving true latitude and longitude for a CRS.

    This is its base geographic CRS if that is in degrees east of Greenwich, so
    no datum shift is applied. Otherwise, e.g. for EPSG:4807 (Paris prime
    meridian, in grads), it is WGS 84.
    """
    base = base_geographic_crs(crs)
    return base if in_degrees(base) and prime_meridian_degrees(base) == 0 else WGS84


def from_lonlat_transformer(crs: CRS) -> Transformer:
    """Return a transformer from true longitude and latitude to a CRS's x and y."""
    return Transformer.from_crs(true_geographic_crs(crs), crs, always_xy=True)


def to_lonlat_transformer(crs: CRS) -> Transformer:
    """Return a transformer from a CRS's x and y to true longitude and latitude."""
    return Transformer.from_crs(crs, true_geographic_crs(crs), always_xy=True)


@dataclass(frozen=True)
class EllipsoidDatum:
    """A CRS's ellipsoid, and the PROJ.4 name of its datum, e.g. "WGS84", if any.

    Converters should keep a named datum, e.g. NAD27, so that cartopy and iris
    shift between it and others, and otherwise use the ellipsoid's axes.
    """

    datum: str | None
    semi_major_axis: float
    semi_minor_axis: float


def ellipsoid_datum(crs: CRS) -> EllipsoidDatum | None:
    """Return the ellipsoid and datum name fully describing a CRS, or None if they can't.

    They describe it when the CRS has a Greenwich prime meridian, and a datum
    that a PROJ.4 string fully describes, either by name, e.g. WGS 84, NAD83 or
    NAD27, or as nothing more than its ellipsoid. Otherwise, e.g. for EPSG:4807
    (Paris prime meridian) or EPSG:4277 (OSGB 1936 datum), the ellipsoid and
    datum name alone would silently drop the prime meridian or datum shift.
    """
    if crs.geodetic_crs is None:
        return None
    geodetic = base_geographic_crs(crs)
    if prime_meridian_degrees(geodetic) != 0 or geodetic.ellipsoid is None:
        return None
    # PROJ.4 strings can only name a few datums, or give a bare ellipsoid
    params = _proj4_dict(geodetic)
    if not geodetic.equals(CRS.from_dict(params), ignore_axis_order=True):
        return None
    return EllipsoidDatum(
        datum=params.get("datum"),
        semi_major_axis=geodetic.ellipsoid.semi_major_metre,
        semi_minor_axis=geodetic.ellipsoid.semi_minor_metre,
    )


def cf_rotation(cf: dict[str, Any]) -> dict[str, float]:
    """Return the pole rotation of a rotated pole CRS, given as its CF parameters.

    The keys are the CF parameter names, as iris's RotatedGeogCS takes them.
    """
    return {
        "grid_north_pole_latitude": cf["grid_north_pole_latitude"],
        "grid_north_pole_longitude": cf["grid_north_pole_longitude"],
        "north_pole_grid_longitude": cf.get("north_pole_grid_longitude", 0.0),
    }


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


def metres_per_unit(crs: CRS) -> float:
    """Return the length of a projected CRS's linear unit, in metres.

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
    return factor


def prime_meridian_degrees(crs: CRS) -> float:
    """Return the longitude of the CRS's prime meridian east of Greenwich, in degrees."""
    pm = crs.prime_meridian
    if pm is None:
        return 0.0
    return math.degrees(pm.longitude * float(pm.unit_conversion_factor))


def _proj4_dict(crs: CRS) -> dict[str, Any]:
    """Return the PROJ.4 parameters of a CRS, without warning that they are lossy."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "You will likely lose", UserWarning)
        return crs.to_dict()
