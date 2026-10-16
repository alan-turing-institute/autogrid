from typing import Any

import numpy as np
import pytest

import autogrid.scattered
from autogrid import Grid
from autogrid.grids import build_ease2
from autogrid.scattered import points_to_points
from autogrid.sphere import EARTH_RADIUS_KM


def _haversine_km(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    lat1, lon1, lat2, lon2 = map(np.deg2rad, (lat1, lon1, lat2, lon2))
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


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
def test_matches_argo_source(
    target: Grid,
    observations: tuple[np.ndarray, np.ndarray, np.ndarray],
    distance_scale_km: float,
) -> None:
    lat, lon, values = observations
    result = points_to_points(
        lat, lon, values, target.latitudes, target.longitudes, distance_scale_km
    )
    assert result.shape == (3, *target.shape)
    for got, variable in zip(result, values, strict=True):
        expected = _argo_source(
            lat, lon, variable, target.latitudes, target.longitudes, distance_scale_km
        )
        np.testing.assert_allclose(got, expected, rtol=1e-9, atol=1e-12)


def test_chunking_does_not_change_result(
    target: Grid,
    observations: tuple[np.ndarray, np.ndarray, np.ndarray],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lat, lon, values = observations
    args = (lat, lon, values, target.latitudes, target.longitudes, 500.0)
    whole = points_to_points(*args)
    # Chunks of 7 target points, so the last chunk is partial
    monkeypatch.setattr(autogrid.scattered, "_CHUNK_ELEMENTS", 7 * lat.size)
    np.testing.assert_allclose(points_to_points(*args), whole, rtol=1e-12, atol=1e-14)


def test_recovers_observations_at_their_positions() -> None:
    lat, lon = np.array([-70.0, -65.0, -75.0]), np.array([0.0, 90.0, -120.0])
    values = np.array([1.0, 2.0, 3.0])
    result = points_to_points(lat, lon, values, lat, lon, distance_scale_km=10.0)
    np.testing.assert_allclose(result, values)


def test_far_points_tend_to_mean_of_observations() -> None:
    lat, lon, values = np.array([60.0, 61.0]), np.array([0.0, 1.0]), [1.0, 3.0]
    result = points_to_points(lat, lon, values, [-60.0], [180.0], distance_scale_km=50)
    np.testing.assert_allclose(result, [2.0])
    zero_floor = points_to_points(
        lat, lon, values, [-60.0], [180.0], distance_scale_km=50, min_weight=0
    )
    assert np.isnan(zero_floor).all()


def test_wraps_across_dateline() -> None:
    # 179 and -179 are 2 degrees apart across the dateline, not 358
    result = points_to_points(
        [0.0, 0.0], [179.0, 0.0], [1.0, 0.0], [0.0], [-179.0], distance_scale_km=500
    )
    np.testing.assert_allclose(result, [1.0])


def test_ignores_nan_observations_per_slice() -> None:
    lat, lon = np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.0, 2.0])
    values = np.array([[1.0, np.nan, 3.0], [1.0, 2.0, 3.0], [np.nan] * 3])
    result = points_to_points(lat, lon, values, lat, lon, distance_scale_km=50.0)
    expected = points_to_points(lat[[0, 2]], lon[[0, 2]], [1.0, 3.0], lat, lon, 50.0)
    np.testing.assert_allclose(result[0], expected)
    assert np.isfinite(result[1]).all()
    assert np.isnan(result[2]).all()


def test_ignores_masked_observations() -> None:
    lat, lon = np.array([0.0, 0.0]), np.array([0.0, 1.0])
    values = np.ma.masked_array([1.0, 100.0], mask=[False, True])
    result = points_to_points(lat, lon, values, lat[:1], lon[:1], 50.0)
    np.testing.assert_allclose(result, [1.0])
    assert not np.ma.isMaskedArray(result)


def test_keeps_source_and_target_shapes(target: Grid) -> None:
    lat, lon = np.meshgrid(np.linspace(-80, -60, 4), np.linspace(-180, 180, 5))
    values = np.ones((2, *lat.shape), dtype=np.float32)
    result = points_to_points(
        lat, lon, values, target.latitudes, target.longitudes, 1e3
    )
    assert result.shape == (2, *target.shape)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, 1.0, rtol=1e-6)


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
def test_rejects_invalid_arguments(kwargs: dict[str, Any], match: str) -> None:
    args: dict[str, Any] = {
        "source_lats": [0.0],
        "source_lons": [0.0],
        "values": [1.0],
        "target_lats": [0.0],
        "target_lons": [0.0],
        "distance_scale_km": 100.0,
    } | kwargs
    with pytest.raises(ValueError, match=match):
        points_to_points(**args)


def test_max_distance_beyond_globe_matches_dense(
    target: Grid, observations: tuple[np.ndarray, np.ndarray, np.ndarray]
) -> None:
    lat, lon, values = observations
    args = (lat, lon, values, target.latitudes, target.longitudes, 500.0)
    np.testing.assert_allclose(
        points_to_points(*args, max_distance_km=25000.0),
        points_to_points(*args),
        rtol=1e-12,
        atol=1e-14,
    )


def test_max_distance_matches_brute_force(
    target: Grid, observations: tuple[np.ndarray, np.ndarray, np.ndarray]
) -> None:
    lat, lon, values = observations
    distance_scale_km, max_distance_km = 300.0, 800.0
    result = points_to_points(
        lat,
        lon,
        values,
        target.latitudes,
        target.longitudes,
        distance_scale_km,
        max_distance_km=max_distance_km,
    )
    distance_km = _haversine_km(
        target.latitudes.ravel()[:, None], target.longitudes.ravel()[:, None], lat, lon
    )
    weights = np.exp(-0.5 * (distance_km / distance_scale_km) ** 2).clip(min=1e-10)
    weights[distance_km > max_distance_km] = 0.0
    with np.errstate(invalid="ignore"):
        expected = (values @ weights.T) / weights.sum(axis=1)
    expected = expected.reshape(3, *target.shape)
    # Some target points have no observations within the cutoff, others do
    assert np.isnan(expected).any() and np.isfinite(expected).any()
    np.testing.assert_array_equal(np.isnan(result), np.isnan(expected))
    np.testing.assert_allclose(result, expected, rtol=1e-9, atol=1e-12)


def test_max_distance_includes_coincident_points() -> None:
    lat, lon = np.array([-70.0, -65.0]), np.array([0.0, 90.0])
    result = points_to_points(
        lat, lon, [1.0, 2.0], lat, lon, distance_scale_km=10.0, max_distance_km=1.0
    )
    np.testing.assert_allclose(result, [1.0, 2.0])


def test_max_distance_leaves_far_points_nan() -> None:
    result = points_to_points(
        [60.0], [0.0], [1.0], [60.0, -60.0], [0.0, 0.0], 50.0, max_distance_km=250.0
    )
    np.testing.assert_array_equal(np.isnan(result), [False, True])


def test_rejects_non_positive_max_distance() -> None:
    with pytest.raises(ValueError, match="max_distance_km must be positive"):
        points_to_points([0.0], [0.0], [1.0], [0.0], [0.0], 100.0, max_distance_km=0)
