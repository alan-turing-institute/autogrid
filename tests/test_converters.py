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
    assert np.isnan(grid.latlon()[0]).any()
    return grid


def test_grid_to_points_matches_nearest_indices(latlon: Grid) -> None:
    rng = np.random.default_rng(0)
    lats, lons = rng.uniform(-90, 90, (4, 5)), rng.uniform(-180, 180, (4, 5))
    data = rng.random((3, *latlon.shape)).astype(np.float32)
    grid_lats, grid_lons = latlon.latlon()
    iy, ix = nearest_indices(
        source_lats=grid_lats, source_lons=grid_lons, target_lats=lats, target_lons=lons
    )
    result = grid_to_points(latlon, data, target_lats=lats, target_lons=lons)
    assert result.shape == (3, 4, 5)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, data[..., iy, ix])


def test_grid_to_points_skips_grid_points_without_latlon() -> None:
    grid = _beyond_projection()
    field = np.arange(grid.size).reshape(grid.shape)
    lat, lon = grid.latlon()
    has_latlon = np.isfinite(lat)
    result = grid_to_points(
        grid, field, target_lats=lat[has_latlon], target_lons=lon[has_latlon]
    )
    np.testing.assert_array_equal(result, field[has_latlon])


def test_grid_to_points_rejects_wrong_data_shape(latlon: Grid) -> None:
    with pytest.raises(ValueError, match="does not end with the grid shape"):
        grid_to_points(latlon, np.ones((3, 4)), target_lats=[0], target_lons=[0])


def test_points_to_grid_matches_gaussian_weighted_mean() -> None:
    grid = build_global_latlon(10.0)
    rng = np.random.default_rng(0)
    lats, lons = rng.uniform(-90, 90, 50), rng.uniform(-180, 180, 50)
    data = rng.normal(size=(2, 50))
    grid_lats, grid_lons = grid.latlon()
    expected = gaussian_weighted_mean(
        source_lats=lats,
        source_lons=lons,
        data=data,
        target_lats=grid_lats,
        target_lons=grid_lons,
        distance_scale_km=500,
    )
    result = points_to_grid(
        grid, source_lats=lats, source_lons=lons, data=data, distance_scale_km=500
    )
    assert result.shape == (2, *grid.shape)
    np.testing.assert_array_equal(result, expected)


def test_points_to_grid_leaves_grid_points_without_latlon_nan() -> None:
    grid = _beyond_projection()
    result = points_to_grid(
        grid,
        source_lats=[80.0, 70.0],
        source_lons=[0.0, 90.0],
        data=[1.0, 2.0],
        distance_scale_km=500,
    )
    no_latlon = np.isnan(grid.latlon()[0])
    np.testing.assert_array_equal(np.isnan(result), no_latlon)
