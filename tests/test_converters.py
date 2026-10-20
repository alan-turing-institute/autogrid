import numpy as np
import pytest

from autogrid import Grid, grid_to_points, points_to_grid
from autogrid.grids import build_global_latlon
from autogrid.points import gaussian_weighted_mean, nearest_indices


def _beyond_projection() -> Grid:
    # A north polar grid reaching beyond the valid area of its projection, so
    # that its corner cells have no latitude or longitude
    points = np.arange(-15_000_000, 15_000_001, 300_000.0)
    grid = Grid(points, points[::-1], "EPSG:6931")
    assert np.isnan(grid.latitudes).any()
    return grid


def test_grid_to_points_matches_nearest_indices(latlon: Grid) -> None:
    rng = np.random.default_rng(0)
    lats, lons = rng.uniform(-90, 90, (4, 5)), rng.uniform(-180, 180, (4, 5))
    data = rng.random((3, *latlon.shape)).astype(np.float32)
    iy, ix = nearest_indices(latlon.latitudes, latlon.longitudes, lats, lons)
    result = grid_to_points(latlon, data, lats, lons)
    assert result.shape == (3, 4, 5)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, data[..., iy, ix])


def test_grid_to_points_skips_grid_points_without_latlon() -> None:
    grid = _beyond_projection()
    field = np.arange(grid.size).reshape(grid.shape)
    has_latlon = np.isfinite(grid.latitudes)
    result = grid_to_points(
        grid, field, grid.latitudes[has_latlon], grid.longitudes[has_latlon]
    )
    np.testing.assert_array_equal(result, field[has_latlon])


def test_grid_to_points_rejects_wrong_data_shape(latlon: Grid) -> None:
    with pytest.raises(ValueError, match="does not end with the grid shape"):
        grid_to_points(latlon, np.ones((3, 4)), [0], [0])


def test_points_to_grid_matches_gaussian_weighted_mean() -> None:
    grid = build_global_latlon(10.0)
    rng = np.random.default_rng(0)
    lats, lons = rng.uniform(-90, 90, 50), rng.uniform(-180, 180, 50)
    values = rng.normal(size=(2, 50))
    expected = gaussian_weighted_mean(
        lats, lons, values, grid.latitudes, grid.longitudes, distance_scale_km=500
    )
    result = points_to_grid(grid, lats, lons, values, distance_scale_km=500)
    assert result.shape == (2, *grid.shape)
    np.testing.assert_array_equal(result, expected)


def test_points_to_grid_leaves_grid_points_without_latlon_nan() -> None:
    grid = _beyond_projection()
    result = points_to_grid(grid, [80.0, 70.0], [0.0, 90.0], [1.0, 2.0], 500)
    no_latlon = np.isnan(grid.latitudes)
    np.testing.assert_array_equal(np.isnan(result), no_latlon)
