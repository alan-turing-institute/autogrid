from typing import Any

import numpy as np
import pytest
from pyproj import CRS

from autogrid import Grid
from autogrid.crs import CRSLike
from autogrid.grids import build_ease2, build_global_latlon

PARIS = "+proj=longlat +ellps=clrk80ign +pm=paris +no_defs"
PARIS_DEGREES = 2.5969213 * 0.9


def test_equal_grids_hash_equal() -> None:
    lon = np.arange(-179.5, 180, 1.0)
    lat = np.arange(-89.5, 90, 1.0)
    cache = {build_global_latlon(1.0): "cached"}
    assert cache[Grid(lon, lat, "EPSG:4326")] == "cached"
    a = Grid([0.0, 1.0], [0.0, 1.0], 4326)
    assert hash(a) == hash(Grid([-0.0, 1.0], [0.0, 1.0], 4326))


def test_equivalent_crs_grids_hash_equal() -> None:
    # Rebuild EPSG:4326 without its identifier or name, so its WKT differs
    crs_dict = CRS.from_epsg(4326).to_json_dict()
    del crs_dict["id"]
    crs_dict["name"] = "renamed"
    a = Grid([0.0, 1.0], [0.0, 1.0], 4326)
    b = Grid([0.0, 1.0], [0.0, 1.0], CRS.from_json_dict(crs_dict))
    assert a == b
    assert hash(a) == hash(b)


def test_grids_differing_only_in_crs_are_distinct_keys() -> None:
    rotated = "+proj=ob_tran +o_proj=longlat +o_lon_p=0 +o_lat_p=37.5 +lon_0=357.5"
    a = Grid([0.0, 1.0], [0.0, 1.0], rotated)
    b = Grid([0.0, 1.0], [0.0, 1.0], PARIS)
    c = Grid([0.0, 1.0], [0.0, 1.0], 4326)
    assert len({a, b, c}) == 3


def test_name_is_ignored_by_equality() -> None:
    a = Grid([0.0, 1.0], [0.0, 1.0], 4326, name="a")
    b = Grid([0.0, 1.0], [0.0, 1.0], 4326, name="b")
    assert a == b
    assert hash(a) == hash(b)


def test_repr_shows_name_when_set() -> None:
    assert repr(Grid([0.0, 1.0], [0.0, 1.0], 4326)) == (
        "Grid(shape=(2, 2), crs='EPSG:4326')"
    )
    assert repr(build_ease2("south", "25p0km", shape=(432, 432))) == (
        "Grid(name='EASE2 south 25p0km', shape=(432, 432), crs='EPSG:6932')"
    )
    assert build_global_latlon(0.25).name == "Global lat-lon 0.25 degree"


def test_grid_points_are_read_only() -> None:
    lon = np.arange(-179.5, 180, 1.0)
    grid = Grid(lon, np.arange(-89.5, 90, 1.0), "EPSG:4326")
    lon[0] = 0.0
    assert grid.x_points[0] == -179.5
    with pytest.raises(ValueError, match="read-only"):
        grid.x_points[0] = 0.0


@pytest.mark.parametrize(
    "grid",
    [
        build_global_latlon(1.0),
        build_ease2("north", "25p0km", shape=(432, 432)),
        build_ease2("global", "36p0km"),
        Grid(
            np.linspace(-5, 5, 20),
            np.linspace(-5, 5, 30),
            CRS.from_cf(
                {
                    "grid_mapping_name": "rotated_latitude_longitude",
                    "grid_north_pole_latitude": 37.5,
                    "grid_north_pole_longitude": 177.5,
                }
            ),
        ),
    ],
    ids=["latlon", "ease2-north", "ease2-global", "rotated"],
)
def test_from_latlon_recovers_grid(grid: Grid) -> None:
    recovered = Grid.from_latlon(grid.latitudes, grid.longitudes, grid.crs)
    assert type(recovered) is Grid
    assert recovered.shape == grid.shape
    np.testing.assert_allclose(recovered.x_points, grid.x_points, atol=1e-6)
    np.testing.assert_allclose(recovered.y_points, grid.y_points, atol=1e-6)
    assert recovered.is_circular == grid.is_circular


def test_from_latlon_handles_dateline() -> None:
    lon, lat = np.meshgrid(np.arange(170.5, 190, 1.0), np.arange(-9.5, 10, 1.0))
    lon = (lon + 180) % 360 - 180  # 170.5 ... 179.5, -179.5 ... -170.5
    grid = Grid.from_latlon(lat, lon, "EPSG:4326")
    np.testing.assert_allclose(grid.x_points, np.arange(170.5, 190, 1.0))


def test_from_latlon_handles_rows_starting_on_different_branches() -> None:
    lon, lat = np.meshgrid(np.arange(180, 190, 1.0), np.arange(-2.5, 3, 1.0))
    lon[1::2, 0] = -180  # Some rows start at 180 and others at -180
    grid = Grid.from_latlon(lat, lon, "EPSG:4326")
    np.testing.assert_allclose(grid.x_points, np.arange(180, 190, 1.0))


def test_from_latlon_rejects_wrong_crs() -> None:
    grid = build_ease2("north", "25p0km", shape=(432, 432))
    with pytest.raises(ValueError, match="not on a rectilinear grid"):
        Grid.from_latlon(grid.latitudes, grid.longitudes, "EPSG:4326")


@pytest.mark.parametrize(
    ("lats", "lons"),
    [(np.zeros(10), np.zeros(10)), (np.zeros((3, 4)), np.zeros((4, 3)))],
    ids=["1d", "mismatched"],
)
def test_from_latlon_rejects_bad_shapes(lats: np.ndarray, lons: np.ndarray) -> None:
    with pytest.raises(ValueError, match="2D arrays of the same shape"):
        Grid.from_latlon(lats, lons, "EPSG:4326")


@pytest.mark.parametrize("shape", [(1, 5), (5, 1), (0, 5)])
def test_from_latlon_rejects_too_few_points(shape: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="At least two rows and two columns"):
        Grid.from_latlon(np.zeros(shape), np.zeros(shape), "EPSG:4326")


@pytest.mark.parametrize(
    ("points", "match"),
    [
        ([5.0], "at least two points"),
        ([], "at least two points"),
        ([[0.0, 1.0], [2.0, 3.0]], "1D array"),
        ([0.0, np.nan, 2.0], "finite"),
        ([0.0, np.inf], "finite"),
        ([0.0, 2.0, 1.0], "strictly increasing or strictly decreasing"),
        ([0.0, 1.0, 1.0], "strictly increasing or strictly decreasing"),
    ],
)
def test_grid_rejects_invalid_points(points: list[Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        Grid(points, [0.0, 1.0], 4326)
    with pytest.raises(ValueError, match=match):
        Grid([0.0, 1.0], points, 4326)


@pytest.mark.parametrize("crs", ["EPSG:6931", 6931, CRS.from_epsg(6931)])
def test_grid_accepts_any_pyproj_crs_input(crs: CRSLike) -> None:
    grid = Grid([-12500.0, 12500.0], [12500.0, -12500.0], crs)
    assert grid == build_ease2("north", "25p0km", shape=(2, 2))


def test_grid_has_no_iris_attributes() -> None:
    grid = build_ease2("south", "25p0km", shape=(432, 432))
    for name, value in vars(grid).items():
        assert not type(value).__module__.startswith("iris"), name


def test_grid_bounds_are_halfway_between_points() -> None:
    grid = Grid([0.0, 1.0, 3.0], [10.0, 20.0], "EPSG:4326")
    np.testing.assert_array_equal(grid.x_bounds, [[-0.5, 0.5], [0.5, 2.0], [2.0, 4.0]])
    np.testing.assert_array_equal(grid.y_bounds, [[5.0, 15.0], [15.0, 25.0]])


def test_grids_with_different_crs_are_not_equal() -> None:
    assert build_ease2("north", "36p0km") != build_ease2("south", "36p0km")


def test_grid_latlon_is_relative_to_greenwich() -> None:
    grid = Grid([0.0, 1.0], [45.0, 46.0], PARIS)
    np.testing.assert_allclose(grid.longitudes[0], [PARIS_DEGREES, PARIS_DEGREES + 1])


def test_grid_latlon_converts_grads_to_degrees() -> None:
    # EPSG:4807 is in grads relative to Paris, so 0 and 50 grads are 2.34 and 45 degrees
    grid = Grid([0.0, 1.0], [50.0, 51.0], "EPSG:4807")
    np.testing.assert_allclose(grid.longitudes[0, 0], PARIS_DEGREES, atol=1e-2)
    np.testing.assert_allclose(grid.latitudes[0, 0], 45.0, atol=1e-2)


def test_rotated_grid_latlon_is_unrotated() -> None:
    crs = CRS.from_cf(
        {
            "grid_mapping_name": "rotated_latitude_longitude",
            "grid_north_pole_latitude": 37.5,
            "grid_north_pole_longitude": 177.5,
        }
    )
    grid = Grid([0.0, 1.0], [0.0, 1.0], crs)
    np.testing.assert_allclose(grid.latitudes[0, 0], 52.5)
    np.testing.assert_allclose(grid.longitudes[0, 0], -2.5)


def test_from_latlon_unwraps_grads_across_dateline() -> None:
    # EPSG:4807 is in grads relative to Paris, so its dateline is at 200 grads
    grid = Grid([198.0, 199.0, 201.0, 202.0], [50.0, 51.0], "EPSG:4807")
    rebuilt = Grid.from_latlon(grid.latitudes, grid.longitudes, "EPSG:4807")
    np.testing.assert_allclose(np.diff(rebuilt.x_points), [1.0, 2.0, 1.0], atol=1e-6)


def test_contains_uses_cell_bounds() -> None:
    grid = Grid([0.0, 1.0, 2.0], [10.0, 11.0], "EPSG:4326")
    x = [-0.5, -0.6, 2.5, 2.6, 1.0, 1.0, 1.0, np.nan]
    y = [10.0, 10.0, 10.0, 10.0, 9.5, 11.6, np.nan, 10.0]
    np.testing.assert_array_equal(
        grid.contains(x, y), [True, False, True, False, True, False, False, False]
    )


def test_contains_wraps_longitude() -> None:
    grid = Grid(np.arange(170.5, 200, 1.0), [0.0, 1.0], "EPSG:4326")
    np.testing.assert_array_equal(
        grid.contains([-170.0, 170.0, 0.0], [0.0, 0.0, 0.0]), [True, True, False]
    )


def test_contains_includes_dateline_gap_of_circular_grid(latlon: Grid) -> None:
    np.testing.assert_array_equal(
        latlon.contains([-179.9, 179.9, 180.0, 540.0, np.inf], [0.0] * 5),
        [True, True, True, True, False],
    )


def test_contains_keeps_projected_shape() -> None:
    grid = build_ease2("south", "25p0km", shape=(432, 432))
    inside = grid.contains(grid.x, grid.y)
    assert inside.shape == grid.shape
    assert inside.all()
    assert not grid.contains([1e8], [0.0]).any()
