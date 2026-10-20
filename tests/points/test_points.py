from typing import Any

import numpy as np
import pytest

import autogrid.points.points
from autogrid import Grid
from autogrid.grids import build_ease2, build_global_latlon
from autogrid.points import gaussian_mean, nearest_indices
from autogrid.points.sphere import EARTH_RADIUS_KM


def _haversine_km(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    lat1, lon1, lat2, lon2 = map(np.deg2rad, (lat1, lon1, lat2, lon2))
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def _curvilinear_points() -> tuple[np.ndarray, np.ndarray]:
    # A lat/lon grid sheared and warped so that rows and columns follow no
    # common lines, so there is no CRS in which it is rectilinear
    i, j = np.meshgrid(np.arange(40), np.arange(30))
    return -80 + 1.5 * j + 0.3 * np.sin(i / 3), -60 + 2.0 * i + 0.5 * j


def test_curvilinear_points_are_not_a_grid() -> None:
    with pytest.raises(ValueError, match="not on a rectilinear grid"):
        Grid.from_latlon(*_curvilinear_points(), "EPSG:4326")


def test_nearest_flat_source_matches_grid_source(latlon: Grid, south: Grid) -> None:
    # An anemoi dataset stores the source as flat latitude and longitude arrays
    (flat,) = nearest_indices(
        latlon.latitudes.ravel(),
        latlon.longitudes.ravel(),
        south.latitudes,
        south.longitudes,
    )
    assert flat.shape == south.shape
    assert flat.dtype == np.intp
    iy, ix = nearest_indices(
        latlon.latitudes, latlon.longitudes, south.latitudes, south.longitudes
    )
    np.testing.assert_array_equal(flat, np.ravel_multi_index((iy, ix), latlon.shape))


def test_nearest_flat_target_keeps_its_shape(latlon: Grid, south: Grid) -> None:
    iy, ix = nearest_indices(
        latlon.latitudes,
        latlon.longitudes,
        south.latitudes.ravel(),
        south.longitudes.ravel(),
    )
    assert iy.shape == ix.shape == (south.latitudes.size,)


@pytest.mark.parametrize("source", ["latlon", "curvilinear"])
def test_nearest_matches_brute_force(source: str) -> None:
    if source == "latlon":
        grid = build_global_latlon(5.0)
        src_lat, src_lon = grid.latitudes, grid.longitudes
    else:
        src_lat, src_lon = _curvilinear_points()
    target = build_ease2("south", "36p0km", shape=(60, 60))
    tgt_lat, tgt_lon = target.latitudes.ravel(), target.longitudes.ravel()
    indices = nearest_indices(src_lat, src_lon, tgt_lat, tgt_lon)
    # Distance from every target point to every source point, as in cryocast
    distances = _haversine_km(
        tgt_lat[:, None], tgt_lon[:, None], src_lat.ravel()[None], src_lon.ravel()[None]
    )
    chosen = _haversine_km(tgt_lat, tgt_lon, src_lat[indices], src_lon[indices])
    # Compare distances rather than indices, as ties may be broken differently
    np.testing.assert_allclose(
        chosen, distances.min(axis=1), rtol=0, atol=1e-12 * EARTH_RADIUS_KM
    )


def test_nearest_wraps_across_dateline_and_longitude_conventions() -> None:
    src_lat = np.array([0.0, 0.0, 0.0])
    src_lon = np.array([179.0, -100.0, 100.0])
    # 181 and -179 are the same longitude, nearest to 179 across the dateline
    (idx,) = nearest_indices(src_lat, src_lon, [0.0, 0.0, 0.0], [181.0, -179.0, 260.0])
    np.testing.assert_array_equal(idx, [0, 0, 1])


@pytest.mark.parametrize(
    ("args", "match"),
    [
        (([0.0, 1.0], [0.0], [0.0], [0.0]), "Source latitudes and longitudes must"),
        (([0.0], [0.0], [0.0], [[0.0]]), "Target latitudes and longitudes must"),
        (([], [], [0.0], [0.0]), "Source points must not be empty"),
        (([0.0, np.nan], [0.0, 1.0], [0.0], [0.0]), "Source .* must all be finite"),
        (([0.0], [0.0], [0.0], [np.inf]), "Target .* must all be finite"),
    ],
    ids=["source-shape", "target-shape", "empty", "source-nan", "target-inf"],
)
def test_nearest_rejects_invalid_points(args: tuple, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        nearest_indices(*args)


def _argo_source(
    obs_lat: np.ndarray,
    obs_lon: np.ndarray,
    values: np.ndarray,
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,
    distance_scale_km: float,
    min_weight: float = 1e-10,
) -> np.ndarray:
    # Dense pairwise weights, as in cryocast's ArgoSource
    distance_km = _haversine_km(
        grid_lat.ravel()[:, None], grid_lon.ravel()[:, None], obs_lat, obs_lon
    )
    weights = np.exp(-0.5 * (distance_km / distance_scale_km) ** 2).clip(min=min_weight)
    return (weights @ values / weights.sum(axis=1)).reshape(grid_lat.shape)


@pytest.fixture(scope="module")
def target() -> Grid:
    return build_ease2("south", "36p0km", shape=(60, 60))


@pytest.fixture(scope="module")
def observations() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    lat = rng.uniform(-80, -60, 50)
    lon = rng.uniform(-180, 180, 50)
    return lat, lon, rng.normal(size=(3, 50))


@pytest.mark.parametrize("distance_scale_km", [100.0, 2000.0])
def test_gaussian_mean_matches_argo_source(
    target: Grid,
    observations: tuple[np.ndarray, np.ndarray, np.ndarray],
    distance_scale_km: float,
) -> None:
    lat, lon, values = observations
    result = gaussian_mean(
        lat, lon, values, target.latitudes, target.longitudes, distance_scale_km
    )
    assert result.shape == (3, *target.shape)
    for got, variable in zip(result, values, strict=True):
        expected = _argo_source(
            lat, lon, variable, target.latitudes, target.longitudes, distance_scale_km
        )
        np.testing.assert_allclose(got, expected, rtol=1e-9, atol=1e-12)


def test_gaussian_mean_chunking_does_not_change_result(
    target: Grid,
    observations: tuple[np.ndarray, np.ndarray, np.ndarray],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lat, lon, values = observations
    args = (lat, lon, values, target.latitudes, target.longitudes, 500.0)
    whole = gaussian_mean(*args)
    # Chunks of 7 target points, so the last chunk is partial
    monkeypatch.setattr(autogrid.points.points, "_CHUNK_ELEMENTS", 7 * lat.size)
    np.testing.assert_allclose(gaussian_mean(*args), whole, rtol=1e-12, atol=1e-14)


def test_gaussian_mean_recovers_observations_at_their_positions() -> None:
    lat, lon = np.array([-70.0, -65.0, -75.0]), np.array([0.0, 90.0, -120.0])
    values = np.array([1.0, 2.0, 3.0])
    result = gaussian_mean(lat, lon, values, lat, lon, distance_scale_km=10.0)
    np.testing.assert_allclose(result, values)


def test_gaussian_mean_far_points_tend_to_mean_of_observations() -> None:
    lat, lon, values = np.array([60.0, 61.0]), np.array([0.0, 1.0]), [1.0, 3.0]
    result = gaussian_mean(lat, lon, values, [-60.0], [180.0], distance_scale_km=50)
    np.testing.assert_allclose(result, [2.0])
    zero_floor = gaussian_mean(
        lat, lon, values, [-60.0], [180.0], distance_scale_km=50, min_weight=0
    )
    assert np.isnan(zero_floor).all()


def test_gaussian_mean_wraps_across_dateline() -> None:
    # 179 and -179 are 2 degrees apart across the dateline, not 358
    result = gaussian_mean(
        [0.0, 0.0], [179.0, 0.0], [1.0, 0.0], [0.0], [-179.0], distance_scale_km=500
    )
    np.testing.assert_allclose(result, [1.0])


def test_gaussian_mean_ignores_nan_observations_per_slice() -> None:
    lat, lon = np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.0, 2.0])
    values = np.array([[1.0, np.nan, 3.0], [1.0, 2.0, 3.0], [np.nan] * 3])
    result = gaussian_mean(lat, lon, values, lat, lon, distance_scale_km=50.0)
    expected = gaussian_mean(lat[[0, 2]], lon[[0, 2]], [1.0, 3.0], lat, lon, 50.0)
    np.testing.assert_allclose(result[0], expected)
    assert np.isfinite(result[1]).all()
    assert np.isnan(result[2]).all()


def test_gaussian_mean_ignores_masked_observations() -> None:
    lat, lon = np.array([0.0, 0.0]), np.array([0.0, 1.0])
    values = np.ma.masked_array([1.0, 100.0], mask=[False, True])
    result = gaussian_mean(lat, lon, values, lat[:1], lon[:1], 50.0)
    np.testing.assert_allclose(result, [1.0])
    assert not np.ma.isMaskedArray(result)


def test_gaussian_mean_keeps_source_and_target_shapes(target: Grid) -> None:
    lat, lon = np.meshgrid(np.linspace(-80, -60, 4), np.linspace(-180, 180, 5))
    values = np.ones((2, *lat.shape), dtype=np.float32)
    result = gaussian_mean(lat, lon, values, target.latitudes, target.longitudes, 1e3)
    assert result.shape == (2, *target.shape)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, 1.0, rtol=1e-6)


def test_gaussian_mean_keeps_scalar_target_shape() -> None:
    result = gaussian_mean([0.0], [0.0], [1.0], 0.0, 0.0, 100.0)
    assert result.shape == ()
    np.testing.assert_allclose(result, 1.0)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"values": [1.0, 2.0]}, "does not end with the source points shape"),
        ({"source_lats": []}, "Source latitudes and longitudes must have"),
        ({"target_lons": [np.nan]}, "Target .* must all be finite"),
        ({"distance_scale_km": 0.0}, "distance_scale_km must be positive"),
        ({"min_weight": -1.0}, "min_weight must not be negative"),
    ],
    ids=["values-shape", "source-shape", "target-nan", "scale", "min-weight"],
)
def test_gaussian_mean_rejects_invalid_arguments(
    kwargs: dict[str, Any], match: str
) -> None:
    args: dict[str, Any] = {
        "source_lats": [0.0],
        "source_lons": [0.0],
        "values": [1.0],
        "target_lats": [0.0],
        "target_lons": [0.0],
        "distance_scale_km": 100.0,
    } | kwargs
    with pytest.raises(ValueError, match=match):
        gaussian_mean(**args)
