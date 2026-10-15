import cartopy.crs as ccrs
import numpy as np
import pytest
from pyproj import CRS

from autogrid import Grid
from autogrid.grid import CRSLike
from autogrid.grids import EASE2Grid, LatLonGrid
from autogrid.grids.ease2 import Region


def test_latlon_grid() -> None:
    grid = LatLonGrid(1.0)
    assert grid.shape == (180, 360)
    assert grid.x_points[0] == -179.5
    assert grid.y_points[-1] == 89.5
    assert grid.is_circular


def test_latlon_grid_rejects_uneven_resolution() -> None:
    with pytest.raises(ValueError, match="does not divide"):
        LatLonGrid(0.7)


def test_latlon_grid_equals_plain_grid() -> None:
    lon = np.arange(-179.5, 180, 1.0)
    lat = np.arange(-89.5, 90, 1.0)
    assert LatLonGrid(1.0) == Grid(lon, lat, "EPSG:4326")


def test_equal_grids_hash_equal() -> None:
    lon = np.arange(-179.5, 180, 1.0)
    lat = np.arange(-89.5, 90, 1.0)
    cache = {LatLonGrid(1.0): "cached"}
    assert cache[Grid(lon, lat, "EPSG:4326")] == "cached"
    assert hash(Grid([0.0], [0.0], 4326)) == hash(Grid([-0.0], [0.0], 4326))


def test_grid_points_are_read_only() -> None:
    lon = np.arange(-179.5, 180, 1.0)
    grid = Grid(lon, np.arange(-89.5, 90, 1.0), "EPSG:4326")
    lon[0] = 0.0
    assert grid.x_points[0] == -179.5
    with pytest.raises(ValueError, match="read-only"):
        grid.x_points[0] = 0.0


@pytest.mark.parametrize(
    ("region", "epsg"), [("north", 6931), ("south", 6932), ("global", 6933)]
)
def test_ease2_crs(region: Region, epsg: int) -> None:
    grid = EASE2Grid(region, "36p0km")
    assert grid.crs.to_epsg() == epsg


def test_ease2_south_25p0km() -> None:
    grid = EASE2Grid("south", "25p0km")
    assert grid.shape == (720, 720)
    assert grid.crs.to_epsg() == 6932
    # Rows run top to bottom, cell centres half a cell in from the edge
    assert grid.x[0, 0] == -8_987_500
    assert grid.y[0, 0] == 8_987_500
    assert np.isclose(grid.latitudes.min(), -90, atol=0.2)


def test_ease2_global_extent() -> None:
    grid = EASE2Grid("global", "36p0km")
    # The global grid spans the whole width of the projection
    x_min, x_max = grid.projection.x_limits
    assert np.isclose(grid.x_bounds[0, 0], x_min)
    assert np.isclose(grid.x_bounds[-1, 1], x_max)


def test_ease2_subset_aligns_with_full_grid() -> None:
    full = EASE2Grid("south", "25p0km")
    subset = EASE2Grid("south", "25p0km", shape=(432, 432))
    offset = (720 - 432) // 2
    np.testing.assert_array_equal(subset.x_points, full.x_points[offset : offset + 432])


@pytest.mark.parametrize("shape", [(431, 431), (800, 800)])
def test_ease2_rejects_misaligned_subset(shape: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="centred subset"):
        EASE2Grid("south", "25p0km", shape=shape)


@pytest.mark.parametrize(
    "grid",
    [
        LatLonGrid(1.0),
        EASE2Grid("north", "25p0km", shape=(432, 432)),
        EASE2Grid("global", "36p0km"),
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


def test_from_latlon_rejects_wrong_crs() -> None:
    grid = EASE2Grid("north", "25p0km", shape=(432, 432))
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


@pytest.mark.parametrize("crs", ["EPSG:6931", 6931, CRS.from_epsg(6931)])
def test_grid_accepts_any_pyproj_crs_input(crs: CRSLike) -> None:
    grid = Grid([-12500.0, 12500.0], [12500.0, -12500.0], crs)
    assert grid == EASE2Grid("north", "25p0km", shape=(2, 2))


def test_grid_has_no_iris_attributes() -> None:
    grid = EASE2Grid("south", "25p0km", shape=(432, 432))
    for name, value in vars(grid).items():
        assert not type(value).__module__.startswith("iris"), name


def test_grid_bounds_are_halfway_between_points() -> None:
    grid = Grid([0.0, 1.0, 3.0], [10.0, 20.0], "EPSG:4326")
    np.testing.assert_array_equal(grid.x_bounds, [[-0.5, 0.5], [0.5, 2.0], [2.0, 4.0]])
    np.testing.assert_array_equal(grid.y_bounds, [[5.0, 15.0], [15.0, 25.0]])


def test_grids_with_different_crs_are_not_equal() -> None:
    assert EASE2Grid("north", "36p0km") != EASE2Grid("south", "36p0km")


@pytest.mark.parametrize("region", ["north", "south", "global"])
def test_ease2_projection_has_plot_limits(region: Region) -> None:
    x_min, x_max = EASE2Grid(region, "36p0km").projection.x_limits
    assert x_max - x_min > 1e7


def test_rotated_grid_projection_is_rotated_pole() -> None:
    crs = CRS.from_cf(
        {
            "grid_mapping_name": "rotated_latitude_longitude",
            "grid_north_pole_latitude": 37.5,
            "grid_north_pole_longitude": 177.5,
        }
    )
    projection = Grid(np.linspace(-5, 5, 20), np.linspace(-5, 5, 30), crs).projection
    assert isinstance(projection, ccrs.RotatedPole)
    assert projection.proj4_params["o_lat_p"] == 37.5


PARIS = "+proj=longlat +ellps=clrk80ign +pm=paris +no_defs"
PARIS_DEGREES = 2.5969213 * 0.9


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


@pytest.mark.parametrize("crs", [PARIS, "EPSG:4807"])
def test_projection_respects_prime_meridian(crs: str) -> None:
    projection = Grid([0.0, 1.0], [45.0, 46.0], crs).projection
    lon, _ = ccrs.PlateCarree().transform_point(0.0, 45.0, projection)
    np.testing.assert_allclose(lon, PARIS_DEGREES, atol=1e-2)


@pytest.mark.parametrize("crs", ["EPSG:4326", "OGC:CRS84", "+proj=longlat +R=6371229"])
def test_plain_latlon_projection_is_plate_carree(crs: str) -> None:
    assert isinstance(Grid([0.0, 1.0], [0.0, 1.0], crs).projection, ccrs.PlateCarree)


def test_projection_keeps_datum() -> None:
    # OSGB 1936 is offset from WGS 84 by about 100 m
    projection = Grid([0.0, 1.0], [52.0, 53.0], "EPSG:4277").projection
    assert not isinstance(projection, ccrs.PlateCarree)
    lon, _ = ccrs.CRS("EPSG:4326").transform_point(-2.0, 52.0, projection)
    assert abs(lon + 2.0) > 1e-3
