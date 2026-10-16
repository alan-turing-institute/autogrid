import numpy as np
import pytest

import autogrid.nearest
from autogrid import Grid
from autogrid.grids import build_ease2, build_global_latlon
from autogrid.nearest import (
    indices_grid_to_grid,
    indices_grid_to_points,
    indices_points_to_points,
)


def _haversine(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    lat1, lon1, lat2, lon2 = map(np.deg2rad, (lat1, lon1, lat2, lon2))
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * np.arcsin(np.sqrt(a))


def _assert_equally_near(
    source: Grid,
    target: Grid,
    got: tuple[np.ndarray, ...],
    want: tuple[np.ndarray, ...],
    max_degrees: float = 180.0,
) -> None:
    # Compare distances rather than indices, as ties may be broken differently
    lat, lon = source.latitudes, source.longitudes
    got_distance, want_distance = (
        _haversine(target.latitudes, target.longitudes, lat[i], lon[i])
        for i in (got, want)
    )
    near = want_distance <= np.deg2rad(max_degrees)
    np.testing.assert_allclose(
        got_distance[near], want_distance[near], rtol=0, atol=1e-12
    )


@pytest.mark.parametrize(
    ("source", "target"),
    [
        (build_global_latlon(1.0), build_ease2("south", "25p0km", shape=(432, 432))),
        (
            Grid(np.arange(-60.5, 0, 1.0), np.arange(-89.5, -40, 1.0), "EPSG:4326"),
            build_ease2("south", "25p0km", shape=(432, 432)),
        ),
        (build_ease2("south", "36p0km", shape=(200, 200)), build_global_latlon(2.0)),
        (build_ease2("north", "36p0km"), build_global_latlon(1.0)),
        (build_ease2("global", "36p0km"), build_global_latlon(1.0)),
        (
            Grid(np.arange(170.5, 200, 1.0), np.arange(60.5, 40, -1.0), "EPSG:4326"),
            build_global_latlon(2.0),
        ),
    ],
    ids=["latlon", "regional", "polar", "polar-full", "cylindrical", "dateline"],
)
def test_grid_search_matches_kd_tree(source: Grid, target: Grid) -> None:
    # Including target points outside the source grid, which are nearest its
    # edge, and the antipode of the polar grids, but not those more than 90
    # degrees from it, for which the edge can be roughly equidistant
    got = indices_grid_to_points(source, target.latitudes, target.longitudes)
    want = indices_points_to_points(
        source.latitudes, source.longitudes, target.latitudes, target.longitudes
    )
    assert got[0].shape == got[1].shape == target.shape
    assert got[0].dtype == got[1].dtype == np.intp
    _assert_equally_near(source, target, got, want, max_degrees=90.0)


def test_grid_search_keeps_target_shape(latlon: Grid) -> None:
    iy, ix = indices_grid_to_points(latlon, [[10.2, -45.2]], [[179.9, -0.4]])
    assert iy.shape == ix.shape == (1, 2)
    np.testing.assert_array_equal(iy, [[100, 44]])
    np.testing.assert_array_equal(ix, [[359, 179]])


def test_grid_search_rejects_invalid_targets(latlon: Grid) -> None:
    with pytest.raises(ValueError, match="Target .* must all be finite"):
        indices_grid_to_points(latlon, [0.0], [np.nan])


def test_flat_source_matches_grid_source(latlon: Grid, south: Grid) -> None:
    # An anemoi dataset stores the source as flat latitude and longitude arrays
    (flat,) = indices_points_to_points(
        latlon.latitudes.ravel(),
        latlon.longitudes.ravel(),
        south.latitudes,
        south.longitudes,
    )
    assert flat.shape == south.shape
    assert flat.dtype == np.intp
    iy, ix = indices_points_to_points(
        latlon.latitudes, latlon.longitudes, south.latitudes, south.longitudes
    )
    np.testing.assert_array_equal(flat, np.ravel_multi_index((iy, ix), latlon.shape))


def test_flat_target_keeps_its_shape(latlon: Grid, south: Grid) -> None:
    iy, ix = indices_points_to_points(
        latlon.latitudes,
        latlon.longitudes,
        south.latitudes.ravel(),
        south.longitudes.ravel(),
    )
    assert iy.shape == ix.shape == (south.latitudes.size,)


def _curvilinear_points() -> tuple[np.ndarray, np.ndarray]:
    # A lat/lon grid sheared and warped so that rows and columns follow no
    # common lines, so there is no CRS in which it is rectilinear
    i, j = np.meshgrid(np.arange(40), np.arange(30))
    return -80 + 1.5 * j + 0.3 * np.sin(i / 3), -60 + 2.0 * i + 0.5 * j


def test_curvilinear_points_are_not_a_grid() -> None:
    with pytest.raises(ValueError, match="not on a rectilinear grid"):
        Grid.from_latlon(*_curvilinear_points(), "EPSG:4326")


@pytest.mark.parametrize("source", ["latlon", "curvilinear"])
def test_matches_brute_force(source: str) -> None:
    if source == "latlon":
        grid = build_global_latlon(5.0)
        src_lat, src_lon = grid.latitudes, grid.longitudes
    else:
        src_lat, src_lon = _curvilinear_points()
    target = build_ease2("south", "36p0km", shape=(60, 60))
    tgt_lat, tgt_lon = target.latitudes.ravel(), target.longitudes.ravel()
    indices = indices_points_to_points(src_lat, src_lon, tgt_lat, tgt_lon)
    # Distance from every target point to every source point, as in cryocast
    distances = _haversine(
        tgt_lat[:, None], tgt_lon[:, None], src_lat.ravel()[None], src_lon.ravel()[None]
    )
    chosen = _haversine(tgt_lat, tgt_lon, src_lat[indices], src_lon[indices])
    # Compare distances rather than indices, as ties may be broken differently
    np.testing.assert_allclose(chosen, distances.min(axis=1), rtol=0, atol=1e-12)


def test_wraps_across_dateline_and_longitude_conventions() -> None:
    src_lat = np.array([0.0, 0.0, 0.0])
    src_lon = np.array([179.0, -100.0, 100.0])
    # 181 and -179 are the same longitude, nearest to 179 across the dateline
    (idx,) = indices_points_to_points(
        src_lat, src_lon, [0.0, 0.0, 0.0], [181.0, -179.0, 260.0]
    )
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
def test_rejects_invalid_points(args: tuple, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        indices_points_to_points(*args)


def test_grid_to_grid_matches_points_in_blocks(
    latlon: Grid, south: Grid, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Blocks of a few rows that do not divide the target rows evenly
    monkeypatch.setattr(autogrid.nearest, "_BLOCK_POINTS", 5 * south.shape[1] + 1)
    got = indices_grid_to_grid(latlon, south)
    want = indices_grid_to_points(latlon, south.latitudes, south.longitudes)
    for g, w in zip(got, want, strict=True):
        np.testing.assert_array_equal(g, w)
