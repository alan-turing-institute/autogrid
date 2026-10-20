"""Helpers for pyproj CRSs, and their cartopy equivalents.

Grids support a small set of CRSs, checked by supported_crs, so that cartopy
and pyproj agree on them, and regridding needs no datum shifts or unit
conversions:

- latitude-longitude or rotated pole CRSs, in degrees;
- projected CRSs, in metres;

each on the WGS 84 datum or on a bare ellipsoid or sphere with no datum, with a
Greenwich prime meridian. Coordinates in any other CRS can be reprojected into
one of these first.

- cartopy: cartopy projections of CRSs, e.g. for plotting map axes. It imports
  cartopy, so is not imported by this package.
"""

from typing import Any

from pyproj import CRS, Transformer

# Anything pyproj.CRS.from_user_input accepts, e.g. "EPSG:6931", 6931 or a CRS
CRSLike = int | str | dict[str, Any] | CRS

GEOGRAPHIC_MAPPINGS = ("latitude_longitude", "rotated_latitude_longitude")


def supported_crs(crs: CRSLike) -> CRS:
    """Return a CRS as a pyproj CRS, checking that Grids support it.

    Raises:
        ValueError: If the CRS is not a 2D latitude-longitude or rotated pole CRS
            in degrees, or projected CRS in metres, on the WGS 84 datum or a bare
            ellipsoid, with a Greenwich prime meridian.
    """
    crs = CRS.from_user_input(crs)
    units = [axis.unit_name for axis in crs.axis_info]
    base = base_geographic_crs(crs) if crs.geodetic_crs is not None else None
    if not (crs.is_geographic or crs.is_projected) or base is None:
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
    elif base.prime_meridian is not None and base.prime_meridian.longitude != 0:
        reason = f"has prime meridian {base.prime_meridian.name}"
    elif not (is_wgs84(crs) or _is_bare_ellipsoid(base)):
        datum = base.datum.name if base.datum is not None else None
        reason = f"has datum {datum!r}"
    else:
        return crs
    msg = (
        f"CRS {crs.name!r} {reason}. Grids support latitude-longitude and rotated "
        "pole CRSs in degrees, and projected CRSs in metres, on the WGS 84 datum or "
        "a bare ellipsoid with a Greenwich prime meridian, so reproject the "
        "coordinates into one of these first."
    )
    raise ValueError(msg)


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


def is_wgs84(crs: CRS) -> bool:
    """Whether a CRS is on the WGS 84 datum, or any of its realisations."""
    datum = base_geographic_crs(crs).datum
    return datum is not None and datum.name.startswith("World Geodetic System 1984")


def ellipsoid_axes(crs: CRS) -> tuple[float, float]:
    """Return the semi-major and semi-minor axes of a CRS's ellipsoid, in metres."""
    ellipsoid = base_geographic_crs(crs).ellipsoid
    if ellipsoid is None:
        msg = f"CRS {crs.name!r} has no ellipsoid."
        raise ValueError(msg)
    return ellipsoid.semi_major_metre, ellipsoid.semi_minor_metre


def _is_bare_ellipsoid(base: CRS) -> bool:
    """Whether a geographic CRS is no more than its ellipsoid, with no datum.

    PROJ names such datums variously, e.g. "unknown", "undefined" or "Not
    specified (based on ...)", but treats them all as equivalent to the bare
    ellipsoid, unlike any named datum or one with a datum shift.
    """
    if base.ellipsoid is None:
        return False
    a, b = ellipsoid_axes(base)
    bare = CRS.from_dict({"proj": "longlat", "a": a, "b": b})
    return base.equals(bare, ignore_axis_order=True)


def from_lonlat_transformer(crs: CRS) -> Transformer:
    """Return a transformer from true longitude and latitude to a CRS's x and y."""
    return Transformer.from_crs(base_geographic_crs(crs), crs, always_xy=True)


def to_lonlat_transformer(crs: CRS) -> Transformer:
    """Return a transformer from a CRS's x and y to true longitude and latitude."""
    return Transformer.from_crs(crs, base_geographic_crs(crs), always_xy=True)
