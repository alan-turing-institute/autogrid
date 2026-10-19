from typing import get_args

import numpy as np
import pytest
from iris.analysis._regrid import RectilinearRegridder
from pyproj import CRS, Transformer

import autogrid.regrid.grid_search
from autogrid import Grid, Regridder
from autogrid.grids import build_global_latlon
from autogrid.regrid.cubes import to_cube
from autogrid.regrid.grid_search import grid_to_grid
from autogrid.regrid.methods import _METHODS, Method


def analytic(grid: Grid) -> np.ndarray:
    return np.cos(np.deg2rad(grid.latitudes)) * np.sin(np.deg2rad(grid.longitudes))


@pytest.mark.parametrize(
    ("method", "atol"),
    [("linear", 1e-3), ("nearest", 2e-2), ("nearest_great_circle", 2e-2)],
)
def test_regrid_matches_analytic_field(
    latlon: Grid, south: Grid, method: Method, atol: float
) -> None:
    result = Regridder(latlon, south).apply(analytic(latlon).astype(np.float32), method)
    assert isinstance(result, np.ndarray)
    assert not isinstance(result, np.ma.MaskedArray)
    assert result.shape == south.shape
    # Within half a cell of the pole, beyond the outermost source latitudes, the
    # source edge values are held rather than extrapolated
    polar = np.abs(south.latitudes) > latlon.y_points.max()
    np.testing.assert_allclose(result[~polar], analytic(south)[~polar], atol=atol)
    np.testing.assert_allclose(result[polar], analytic(south)[polar], atol=1e-2)


@pytest.mark.parametrize("method", ["linear", "nearest"])
def test_rotated_pole_regrids_with_iris(method: Method) -> None:
    rotated = Grid(
        np.arange(-10, 10, 0.5),
        np.arange(-10, 10, 0.5),
        CRS.from_cf(
            {
                "grid_mapping_name": "rotated_latitude_longitude",
                "grid_north_pole_latitude": 37.5,
                "grid_north_pole_longitude": 177.5,
            }
        ),
    )
    target = Grid(np.arange(-5, 5, 1.0), np.arange(45, 55, 1.0), "EPSG:4326")
    result = Regridder(rotated, target).apply(analytic(rotated), method)
    expected = Regridder(rotated, target).apply(
        analytic(rotated), "nearest_great_circle"
    )
    np.testing.assert_allclose(result, analytic(target), atol=5e-3)
    if method == "nearest":
        np.testing.assert_array_equal(result, expected)


def test_area_weighted_conserves_mean(latlon: Grid) -> None:
    data = np.random.default_rng(0).random(latlon.shape)
    result = Regridder(latlon, build_global_latlon(2.0)).apply(data, "area_weighted")
    assert result.shape == (90, 180)
    weights = np.cos(np.deg2rad(latlon.latitudes))
    coarse_weights = np.cos(np.deg2rad(build_global_latlon(2.0).latitudes))
    np.testing.assert_allclose(
        np.average(result, weights=coarse_weights),
        np.average(data, weights=weights),
        rtol=1e-3,
    )


def test_regridder_handles_leading_dimensions(latlon: Grid, south: Grid) -> None:
    regridder = Regridder(latlon, south)
    data = np.random.default_rng(0).random((4, 2, *latlon.shape)).astype(np.float32)
    result = regridder.apply(data)
    assert result.shape == (4, 2, *south.shape)
    np.testing.assert_array_equal(result[3, 1], regridder.apply(data[3, 1]))


def test_points_outside_source_are_nan(south: Grid) -> None:
    regional = Grid(np.arange(-60.5, 0, 1.0), np.arange(-89.5, -40, 1.0), "EPSG:4326")
    result = Regridder(regional, south).apply(np.ones(regional.shape))
    assert np.isnan(result).any()
    np.testing.assert_allclose(result[~np.isnan(result)], 1.0)


@pytest.mark.parametrize("method", ["linear", "nearest"])
def test_half_cell_beyond_outermost_points_keeps_edge_values(method: Method) -> None:
    source = Grid(np.arange(10.0, 20.0, 1.0), np.arange(50.0, 55.0, 1.0), "EPSG:4326")
    data = np.broadcast_to(source.x_points, source.shape) * 10
    # Points just outside the source cells, and inside the edge cells beyond
    # the outermost source points
    target = Grid([9.4, 9.6, 19.4, 19.6], [52.0, 53.0], "EPSG:4326")
    result = Regridder(source, target).apply(data, method)
    np.testing.assert_array_equal(result[:, 1:3], [[100.0, 190.0]] * 2)
    assert np.isnan(result[:, [0, 3]]).all()


def test_integer_input_is_regridded_to_float(south: Grid) -> None:
    regional = Grid(np.arange(-60.5, 0, 1.0), np.arange(-89.5, -40, 1.0), "EPSG:4326")
    regridder = Regridder(regional, south)
    data = np.ones(regional.shape, dtype=np.int32)
    methods: list[Method] = ["linear", "nearest", "nearest_great_circle"]
    for method in methods:
        result = regridder.apply(data, method)
        assert np.issubdtype(result.dtype, np.floating)
        np.testing.assert_array_equal(np.isnan(result), regridder.outside(method))


def test_nan_input_propagates(latlon: Grid, south: Grid) -> None:
    data = np.ones(latlon.shape)
    data[:30] = np.nan  # southernmost 30 degrees
    result = Regridder(latlon, south).apply(data)
    assert np.isnan(result[south.latitudes < -61]).all()
    assert not np.isnan(result[south.latitudes > -59]).any()


def test_rejects_wrong_shape(latlon: Grid, south: Grid) -> None:
    with pytest.raises(ValueError, match="does not end with the source grid shape"):
        Regridder(latlon, south).apply(np.ones((10, 10)))


def test_rejects_unknown_method(latlon: Grid, south: Grid) -> None:
    with pytest.raises(ValueError, match="Unknown regridding method"):
        Regridder(latlon, south).apply(np.ones(latlon.shape), "cubic")  # ty: ignore[invalid-argument-type]


def test_regridder_caches_one_regridder_per_method(latlon: Grid, south: Grid) -> None:
    regridder = Regridder(latlon, south)
    data = np.ones(latlon.shape)
    regridder.apply(data, "linear")
    linear = regridder._methods["linear"]
    regridder.apply(data, "linear")
    regridder.apply(data, "nearest")
    assert set(regridder._methods) == {"linear", "nearest"}
    assert regridder._methods["linear"] is linear


def test_methods_differ_on_same_regridder(latlon: Grid, south: Grid) -> None:
    regridder = Regridder(latlon, south)
    data = analytic(latlon)
    assert not np.array_equal(
        regridder.apply(data, "linear"), regridder.apply(data, "nearest")
    )


@pytest.mark.parametrize(
    "source",
    [
        build_global_latlon(1.0),
        Grid(np.arange(-60.5, 0, 1.0), np.arange(-89.5, -40, 1.0), "EPSG:4326"),
    ],
    ids=["global", "regional"],
)
def test_nearest_great_circle_matches_grid_to_grid_indices(
    source: Grid, south: Grid
) -> None:
    # The indices from grid_to_grid can gather data externally, e.g. torch tensors
    regridder = Regridder(source, south)
    iy, ix = grid_to_grid(source, south)
    data = np.random.default_rng(0).random((3, *source.shape))
    gathered = data[..., iy, ix]
    expected = regridder.apply(data, "nearest_great_circle")
    inside = ~regridder.outside("nearest_great_circle")
    np.testing.assert_array_equal(gathered[..., inside], expected[..., inside])
    assert np.isnan(expected[..., ~inside]).all()


def test_nearest_great_circle_indices_are_cached(
    latlon: Grid, south: Grid, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    real = autogrid.regrid.grid_search.search_grid
    monkeypatch.setattr(
        autogrid.regrid.grid_search,
        "search_grid",
        lambda *args: calls.append(args) or real(*args),
    )
    regridder = Regridder(latlon, south)
    data = np.random.default_rng(0).random(latlon.shape)
    first = regridder.apply(data, "nearest_great_circle")
    np.testing.assert_array_equal(regridder.apply(data, "nearest_great_circle"), first)
    assert len(calls) == 1


def test_nearest_great_circle_differs_from_nearest_near_pole(
    latlon: Grid, south: Grid
) -> None:
    regridder = Regridder(latlon, south)
    flat = np.arange(latlon.x.size, dtype=np.float64).reshape(latlon.shape)
    differ = regridder.apply(flat, "nearest") != regridder.apply(
        flat, "nearest_great_circle"
    )
    assert differ.any()
    assert south.latitudes[differ].min() < -89


def test_nearest_great_circle_handles_leading_dimensions_and_nan(
    latlon: Grid, south: Grid
) -> None:
    regridder = Regridder(latlon, south)
    data = np.random.default_rng(0).random((4, 2, *latlon.shape))
    data[..., :30, :] = np.nan  # southernmost 30 degrees
    result = regridder.apply(data, "nearest_great_circle")
    assert result.shape == (4, 2, *south.shape)
    np.testing.assert_array_equal(
        result[3, 1], regridder.apply(data[3, 1], "nearest_great_circle")
    )
    assert np.isnan(result[..., south.latitudes < -61]).all()
    assert not np.isnan(result[..., south.latitudes > -59]).any()


def test_nearest_great_circle_outside_source_is_nan(south: Grid) -> None:
    regional = Grid(np.arange(-60.5, 0, 1.0), np.arange(-89.5, -40, 1.0), "EPSG:4326")
    result = Regridder(regional, south).apply(
        np.ones(regional.shape), "nearest_great_circle"
    )
    assert np.isnan(result).any()
    np.testing.assert_allclose(result[~np.isnan(result)], 1.0)


def test_outside_matches_iris_sample_points() -> None:
    # UTM points straddling the western edge of a latitude-longitude source,
    # 20 m apart and none exactly on it
    source = Grid(np.arange(-2.0, 0.01, 0.01), np.arange(52.0, 53.01, 0.01), 4326)
    edge = source.x_bounds[0, 0]
    x0, y0 = Transformer.from_crs(4326, 32630, always_xy=True).transform(edge, 52.5)
    target = Grid(x0 + np.arange(-490, 500, 20.0), [y0, y0 + 20], 32630)
    regridder = Regridder(source, target)
    src_y, _ = to_cube(source).dim_coords
    tgt_y, tgt_x = to_cube(target).dim_coords
    x, y = RectilinearRegridder._sample_grid(src_y.coord_system, tgt_x, tgt_y)
    want = ~source.contains(x, y)
    np.testing.assert_array_equal(regridder.outside("linear"), want)
    assert want.any()
    assert not want.all()
    # With no datum shift, the great circle search flags the same points
    np.testing.assert_array_equal(regridder.outside("nearest_great_circle"), want)
    nan = np.isnan(regridder.apply(np.ones(source.shape), "nearest_great_circle"))
    np.testing.assert_array_equal(nan, want)


def test_every_method_is_registered() -> None:
    assert set(_METHODS) == set(get_args(Method))


@pytest.mark.parametrize("method", get_args(Method))
def test_outside_matches_nan_for_each_method(method: Method) -> None:
    # A lat-lon target, as area weighting requires the source coordinate system
    regional = Grid(np.arange(-60.5, 0, 1.0), np.arange(-89.5, -40, 1.0), "EPSG:4326")
    regridder = Regridder(regional, build_global_latlon(2.0))
    result = regridder.apply(np.ones(regional.shape), method)
    outside = regridder.outside(method)
    np.testing.assert_array_equal(np.isnan(result), outside)
    assert outside.any()
    assert not outside.all()
