import numpy as np
import pytest
from pyproj import Transformer

import autogrid.regrid.grid_search
from autogrid import Grid
from autogrid.grids import build_ease2, build_global_latlon
from autogrid.regrid.grid_search import grid_to_grid, grid_to_points, outside_grid
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
) -> None:
    # Compare distances rather than indices, as ties may be broken differently
    lat, lon = source.latitudes, source.longitudes
    got_distance, want_distance = (
        _haversine(target.latitudes, target.longitudes, lat[i], lon[i])
        for i in (got, want)
    )
    np.testing.assert_allclose(got_distance, want_distance, rtol=0, atol=1e-12)


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
    # edge, and the antipode of the polar grids
    got = grid_to_points(source, target.latitudes, target.longitudes)
    want = nearest_indices(
        source.latitudes, source.longitudes, target.latitudes, target.longitudes
    )
    assert got[0].shape == got[1].shape == target.shape
    assert got[0].dtype == got[1].dtype == np.intp
    _assert_equally_near(source, target, got, want)


def test_grid_to_points_finds_nearest_corner_from_outside() -> None:
    # A strip of the northern EASE grid, from which target points far to the
    # south are nearest one end, though they project nearer the other
    ease = build_ease2("north", "36p0km")
    source = Grid(ease.x_points[100:367], ease.y_points[:27], ease.crs)
    target = build_global_latlon(2.0)
    got = grid_to_points(source, target.latitudes, target.longitudes)
    want = nearest_indices(
        source.latitudes, source.longitudes, target.latitudes, target.longitudes
    )
    _assert_equally_near(source, target, got, want)


def test_grid_to_points_keeps_target_shape(latlon: Grid) -> None:
    iy, ix = grid_to_points(latlon, [[10.2, -45.2]], [[179.9, -0.4]])
    assert iy.shape == ix.shape == (1, 2)
    np.testing.assert_array_equal(iy, [[100, 44]])
    np.testing.assert_array_equal(ix, [[359, 179]])


def test_grid_to_points_handles_tiny_source() -> None:
    # Too few points for a quarter-grid step from targets outside it
    source = Grid([0, 1, 2], [0, 1], 4326)
    iy, ix = grid_to_points(source, [0.2, 0.9], [10.0, -5.0])
    np.testing.assert_array_equal(iy, [0, 1])
    np.testing.assert_array_equal(ix, [2, 0])


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


def test_outside_grid_matches_in_blocks(
    south: Grid, monkeypatch: pytest.MonkeyPatch
) -> None:
    regional = Grid(np.arange(-60.5, 0, 1.0), np.arange(-89.5, -40, 1.0), "EPSG:4326")
    want = outside_grid(regional, south)
    monkeypatch.setattr(
        autogrid.regrid.grid_search, "_BLOCK_POINTS", 5 * south.shape[1] + 1
    )
    np.testing.assert_array_equal(outside_grid(regional, south), want)
    assert want.any()
    assert not want.all()


def test_outside_grid_uses_search_coordinates() -> None:
    # British National Grid points (OSGB36 datum) straddling the western edge
    # of a WGS 84 source, 20 m apart, so a datum shift of ~100 m would move
    # the edge by several points
    source = Grid(np.arange(-2.0, 0.01, 0.01), np.arange(52.0, 53.01, 0.01), 4326)
    edge = source.x_bounds[0, 0]
    x0, y0 = Transformer.from_crs(4326, 27700, always_xy=True).transform(edge, 52.5)
    target = Grid(x0 + np.arange(-500, 500, 20.0), [y0, y0 + 20], 27700)
    lats, lons = target.latitudes, target.longitudes
    want = ~source.contains(*source.trf_from_latlon.transform(lons, lats))
    np.testing.assert_array_equal(outside_grid(source, target), want)
    shifted = Transformer.from_crs(27700, 4326, always_xy=True).transform(
        target.x, target.y
    )
    assert (want != ~source.contains(*shifted)).any()
