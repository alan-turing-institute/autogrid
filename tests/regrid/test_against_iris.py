"""Check the regridding weights against iris, which is only a dev dependency."""

import numpy as np
import pytest
from iris.analysis import AreaWeighted, Linear, Nearest
from iris.coord_systems import (
    CoordSystem,
    GeogCS,
    LambertAzimuthalEqualArea,
    RotatedGeogCS,
)
from iris.coords import DimCoord
from iris.cube import Cube
from pyproj import CRS

from autogrid import Grid, Regridder
from autogrid.grids import build_ease2, build_global_latlon
from autogrid.regrid import Method

ROTATED = CRS.from_cf(
    {
        "grid_mapping_name": "rotated_latitude_longitude",
        "grid_north_pole_latitude": 37.5,
        "grid_north_pole_longitude": 177.5,
    }
)


def coord_system(crs: CRS) -> CoordSystem:
    """The iris coordinate system of a WGS 84 CRS used by these tests."""
    wgs84 = GeogCS.from_datum("WGS84")
    cf = crs.to_cf()
    match cf["grid_mapping_name"]:
        case "latitude_longitude":
            return wgs84
        case "rotated_latitude_longitude":
            return RotatedGeogCS(
                cf["grid_north_pole_latitude"],
                cf["grid_north_pole_longitude"],
                ellipsoid=wgs84,
            )
        case "lambert_azimuthal_equal_area":
            return LambertAzimuthalEqualArea(
                cf["latitude_of_projection_origin"],
                cf["longitude_of_projection_origin"],
                cf["false_easting"],
                cf["false_northing"],
                ellipsoid=wgs84,
            )
    raise NotImplementedError(cf["grid_mapping_name"])


def to_cube(grid: Grid, data: np.ndarray | None = None) -> Cube:
    """An iris cube on a Grid, as the Regridder built before dropping iris."""
    cs = coord_system(grid.crs)
    if isinstance(cs, RotatedGeogCS):
        names, units = ("grid_latitude", "grid_longitude"), "degrees"
    elif grid.is_geographic:
        names, units = ("latitude", "longitude"), "degrees"
    else:
        names, units = ("projection_y_coordinate", "projection_x_coordinate"), "m"
    y = DimCoord(
        grid.y_points, names[0], units=units, bounds=grid.y_bounds, coord_system=cs
    )
    x = DimCoord(
        grid.x_points,
        names[1],
        units=units,
        bounds=grid.x_bounds,
        coord_system=cs,
        circular=grid.is_circular,
    )
    if data is None:
        data = np.ma.zeros(grid.shape)
    return Cube(data, dim_coords_and_dims=[(y, 0), (x, 1)])


def iris_regrid(
    scheme: Linear | Nearest | AreaWeighted,
    source: Grid,
    target: Grid,
    data: np.ndarray,
) -> np.ma.MaskedArray:
    regridder = scheme.regridder(to_cube(source), to_cube(target))
    return np.ma.asarray(regridder(to_cube(source, data)).data)


def random_field(grid: Grid, nan_fraction: float = 0.01) -> np.ma.MaskedArray:
    rng = np.random.default_rng(0)
    return np.ma.MaskedArray(
        rng.random(grid.shape), mask=rng.random(grid.shape) < nan_fraction
    )


PAIRS = {
    "latlon-to-ease2": (
        build_global_latlon(1.0),
        build_ease2("south", "25p0km", shape=(432, 432)),
    ),
    "ease2-to-latlon": (
        build_ease2("north", "25p0km", shape=(200, 200)),
        Grid(np.arange(-179.5, 180, 1.0), np.arange(60.25, 90, 0.5), "EPSG:4326"),
    ),
    "rotated-to-latlon": (
        Grid(np.arange(-10, 10, 0.5), np.arange(-10, 10, 0.5), ROTATED),
        Grid(np.arange(-5, 5, 0.3), np.arange(45, 55, 0.3), "EPSG:4326"),
    ),
}


@pytest.mark.parametrize("pair", PAIRS)
@pytest.mark.parametrize(
    ("method", "scheme"),
    [
        ("linear", Linear(extrapolation_mode="mask")),
        ("nearest", Nearest(extrapolation_mode="mask")),
    ],
)
def test_point_methods_match_iris(
    pair: str, method: Method, scheme: Linear | Nearest
) -> None:
    source, target = PAIRS[pair]
    data = random_field(source)
    result = Regridder(source, target).apply(data, method)
    # Iris masks beyond the outermost source points, where the Regridder holds
    # the edge values out to the edge of the source cells, so compare inside them
    inside = ~np.ma.getmaskarray(iris_regrid(scheme, source, target, data.data))
    assert inside.mean() > 0.5
    expected = np.ma.filled(iris_regrid(scheme, source, target, data), np.nan)
    np.testing.assert_allclose(result[inside], expected[inside], rtol=1e-6)
    # NaN source points made some target points NaN
    assert np.isnan(expected[inside]).any()


AREA_PAIRS = {
    "latlon-coarsen": (build_global_latlon(1.0), build_global_latlon(2.5)),
    "latlon-across-dateline": (
        build_global_latlon(1.0),
        Grid(np.arange(0.4, 360, 1.3), np.arange(-60.1, 60, 1.7), "EPSG:4326"),
    ),
    "regional-latlon-partly-outside": (
        Grid(np.arange(-60.5, 0, 1.0), np.arange(-89.5, -40, 1.0), "EPSG:4326"),
        # Iris does not clip cell bounds to the poles, so keep within them
        Grid(np.arange(-90, 30, 3.0), np.arange(-88.5, -30, 3.0), "EPSG:4326"),
    ),
    "ease2": (
        build_ease2("south", "25p0km", shape=(432, 432)),
        build_ease2("south", "36p0km", shape=(320, 320)),
    ),
    "rotated": (
        Grid(np.arange(-10, 10, 0.5), np.arange(-10, 10, 0.5), ROTATED),
        Grid(np.arange(-8.1, 8, 0.7), np.arange(-12.1, 12, 0.7), ROTATED),
    ),
}


@pytest.mark.parametrize("pair", AREA_PAIRS)
def test_area_weighted_matches_iris(pair: str) -> None:
    source, target = AREA_PAIRS[pair]
    data = random_field(source)
    result = Regridder(source, target).apply(data, "area_weighted")
    # mdtol=0 makes NaN any target cell with a NaN source cell, as the Regridder does
    expected = iris_regrid(AreaWeighted(mdtol=0), source, target, data)
    np.testing.assert_allclose(result, np.ma.filled(expected, np.nan), rtol=1e-6)
    assert not np.isnan(result).all()
