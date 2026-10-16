import numpy as np
import pytest

import autogrid.regrid.grid_search
from autogrid import Grid
from autogrid.grids import build_ease2, build_global_latlon
from autogrid.regrid.grid_search import grid_to_grid, grid_to_points
from autogrid.regrid.points import nearest_indices


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
def test_grid_to_points_matches_kd_tree(source: Grid, target: Grid) -> None:
    # Including target points outside the source grid, which are nearest its
    # edge, and the antipode of the polar grids, but not those more than 90
    # degrees from it, for which the edge can be roughly equidistant
    got = grid_to_points(source, target.latitudes, target.longitudes)
    want = nearest_indices(
        source.latitudes, source.longitudes, target.latitudes, target.longitudes
    )
    assert got[0].shape == got[1].shape == target.shape
    assert got[0].dtype == got[1].dtype == np.intp
    _assert_equally_near(source, target, got, want, max_degrees=90.0)


def test_grid_to_points_keeps_target_shape(latlon: Grid) -> None:
    iy, ix = grid_to_points(latlon, [[10.2, -45.2]], [[179.9, -0.4]])
    assert iy.shape == ix.shape == (1, 2)
    np.testing.assert_array_equal(iy, [[100, 44]])
    np.testing.assert_array_equal(ix, [[359, 179]])


def test_grid_to_points_rejects_invalid_targets(latlon: Grid) -> None:
    with pytest.raises(ValueError, match="Target .* must all be finite"):
        grid_to_points(latlon, [0.0], [np.nan])


def test_grid_to_grid_matches_points_in_blocks(
    latlon: Grid, south: Grid, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Blocks of a few rows that do not divide the target rows evenly
    monkeypatch.setattr(
        autogrid.regrid.grid_search, "_BLOCK_POINTS", 5 * south.shape[1] + 1
    )
    got = grid_to_grid(latlon, south)
    want = grid_to_points(latlon, south.latitudes, south.longitudes)
    for g, w in zip(got, want, strict=True):
        np.testing.assert_array_equal(g, w)
