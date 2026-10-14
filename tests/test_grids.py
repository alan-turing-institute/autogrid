import numpy as np
import pytest
from iris.coord_systems import LambertAzimuthalEqualArea

from autogrid import EASE2Grid, Grid, LatLonGrid
from autogrid.coord_systems import WGS84, EASE2GlobalCS


def test_latlon_grid() -> None:
    grid = LatLonGrid(1.0)
    assert grid.shape == (180, 360)
    assert grid.x_coord.points[0] == -179.5
    assert grid.y_coord.points[-1] == 89.5
    assert grid.x_coord.circular


def test_latlon_grid_rejects_uneven_resolution() -> None:
    with pytest.raises(ValueError, match="does not divide"):
        LatLonGrid(0.7)


def test_latlon_grid_equals_plain_grid() -> None:
    lon = np.arange(-179.5, 180, 1.0)
    lat = np.arange(-89.5, 90, 1.0)
    assert LatLonGrid(1.0) == Grid(lon, lat, WGS84)


@pytest.mark.parametrize(
    ("region", "epsg"), [("north", 6931), ("south", 6932), ("global", 6933)]
)
def test_ease2_coord_system(region: str, epsg: int) -> None:
    grid = EASE2Grid(region, "36km")
    assert grid.crs.to_epsg() == epsg


def test_ease2_south_25km() -> None:
    grid = EASE2Grid("south", "25km")
    assert grid.shape == (720, 720)
    assert isinstance(grid.coord_system, LambertAzimuthalEqualArea)
    # Rows run top to bottom, cell centres half a cell in from the edge
    assert grid.x[0, 0] == -8_987_500
    assert grid.y[0, 0] == 8_987_500
    assert np.isclose(grid.latitudes.min(), -90, atol=0.2)


def test_ease2_global_extent() -> None:
    grid = EASE2Grid("global", "36km")
    assert isinstance(grid.coord_system, EASE2GlobalCS)
    x_min, x_max = grid.crs.x_limits
    assert np.isclose(grid.x_coord.bounds[0, 0], x_min)
    assert np.isclose(grid.x_coord.bounds[-1, 1], x_max)


def test_ease2_subset_aligns_with_full_grid() -> None:
    full = EASE2Grid("south", "25km")
    subset = EASE2Grid("south", "25km", shape=(432, 432))
    offset = (720 - 432) // 2
    np.testing.assert_array_equal(
        subset.x_coord.points, full.x_coord.points[offset : offset + 432]
    )


@pytest.mark.parametrize("shape", [(431, 431), (800, 800)])
def test_ease2_rejects_misaligned_subset(shape: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="centred subset"):
        EASE2Grid("south", "25km", shape=shape)
