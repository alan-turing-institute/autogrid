from typing import get_args

import numpy as np
import pytest
from pyproj import CRS, Transformer
from scipy import sparse

import autogrid.regrid.methods
from autogrid import Grid, Regridder, grid_to_points
from autogrid.grids import build_ease2, build_global_latlon
from autogrid.locations import Locations
from autogrid.points import Points, nearest_indices
from autogrid.regrid.methods import METHODS, Method


def analytic(grid: Grid) -> np.ndarray:
    lat, lon = grid.latlon()
    return np.cos(np.deg2rad(lat)) * np.sin(np.deg2rad(lon))


def outside(regridder: Regridder) -> np.ndarray:
    """Flag the target points left NaN for a source field with no NaN values."""
    return np.isnan(regridder.apply(np.ones(regridder.source.shape)))


@pytest.mark.parametrize(
    ("method", "atol"),
    [("linear", 1e-3), ("nearest", 2e-2), ("nearest_great_circle", 2e-2)],
)
def test_regrid_matches_analytic_field(
    latlon: Grid, south: Grid, method: Method, atol: float
) -> None:
    result = Regridder(latlon, south, method).apply(analytic(latlon).astype(np.float32))
    assert isinstance(result, np.ndarray)
    assert not isinstance(result, np.ma.MaskedArray)
    assert result.shape == south.shape
    # Within half a cell of the pole, beyond the outermost source latitudes, the
    # source edge values are held rather than extrapolated
    polar = np.abs(south.latlon()[0]) > latlon.y_points.max()
    np.testing.assert_allclose(result[~polar], analytic(south)[~polar], atol=atol)
    np.testing.assert_allclose(result[polar], analytic(south)[polar], atol=1e-2)


@pytest.mark.parametrize("method", ["linear", "nearest"])
def test_rotated_pole_regrids(method: Method) -> None:
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
    result = Regridder(rotated, target, method).apply(analytic(rotated))
    expected = Regridder(rotated, target, "nearest_great_circle").apply(
        analytic(rotated)
    )
    np.testing.assert_allclose(result, analytic(target), atol=5e-3)
    if method == "nearest":
        np.testing.assert_array_equal(result, expected)


def test_area_weighted_conserves_mean(latlon: Grid) -> None:
    data = np.random.default_rng(0).random(latlon.shape)
    regridder = Regridder(latlon, build_global_latlon(2.0), "area_weighted")
    result = regridder.apply(data)
    assert result.shape == (90, 180)
    weights = np.cos(np.deg2rad(latlon.latlon()[0]))
    coarse_weights = np.cos(np.deg2rad(build_global_latlon(2.0).latlon()[0]))
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
    result = Regridder(source, target, method).apply(data)
    np.testing.assert_array_equal(result[:, 1:3], [[100.0, 190.0]] * 2)
    assert np.isnan(result[:, [0, 3]]).all()


def test_integer_input_is_regridded_to_float(south: Grid) -> None:
    regional = Grid(np.arange(-60.5, 0, 1.0), np.arange(-89.5, -40, 1.0), "EPSG:4326")
    data = np.ones(regional.shape, dtype=np.int32)
    methods: list[Method] = ["linear", "nearest", "nearest_great_circle"]
    for method in methods:
        regridder = Regridder(regional, south, method)
        result = regridder.apply(data)
        assert np.issubdtype(result.dtype, np.floating)
        np.testing.assert_array_equal(np.isnan(result), outside(regridder))


def test_nan_input_propagates(latlon: Grid, south: Grid) -> None:
    data = np.ones(latlon.shape)
    data[:30] = np.nan  # southernmost 30 degrees
    result = Regridder(latlon, south).apply(data)
    lat, _ = south.latlon()
    assert np.isnan(result[lat < -61]).all()
    assert not np.isnan(result[lat > -59]).any()


def test_rejects_wrong_shape(latlon: Grid, south: Grid) -> None:
    with pytest.raises(ValueError, match="does not end with the source shape"):
        Regridder(latlon, south).apply(np.ones((10, 10)))


def test_rejects_unknown_method(latlon: Grid, south: Grid) -> None:
    with pytest.raises(ValueError, match="Unknown regridding method"):
        Regridder(latlon, south, "cubic")  # ty: ignore[invalid-argument-type]


def test_accepts_weights_function(latlon: Grid, south: Grid) -> None:
    def mean_of_linear_and_nearest(
        source: Locations, target: Locations
    ) -> sparse.csr_array:
        return (
            METHODS["linear"](source, target) + METHODS["nearest"](source, target)
        ) / 2

    data = analytic(latlon)
    result = Regridder(latlon, south, mean_of_linear_and_nearest).apply(data)
    linear = Regridder(latlon, south, "linear").apply(data)
    nearest = Regridder(latlon, south, "nearest").apply(data)
    np.testing.assert_allclose(result, (linear + nearest) / 2)


def test_weights_function_explicit_zeros_leave_outside_points_nan(south: Grid) -> None:
    def nearest_with_explicit_zeros(
        source: Locations, target: Locations
    ) -> sparse.csr_array:
        # Add an explicit zero weight from the first source point to every target
        coo = METHODS["nearest"](source, target).tocoo()
        rows = np.concatenate([coo.row, np.arange(target.size)])
        cols = np.concatenate([coo.col, np.zeros(target.size, dtype=int)])
        data = np.concatenate([coo.data, np.zeros(target.size)])
        return sparse.csr_array((data, (rows, cols)), shape=coo.shape)

    regional = Grid(np.arange(-60.5, 0, 1.0), np.arange(-89.5, -40, 1.0), "EPSG:4326")
    regridder = Regridder(regional, south, nearest_with_explicit_zeros)
    expected = outside(Regridder(regional, south, "nearest"))
    assert expected.any()
    np.testing.assert_array_equal(outside(regridder), expected)


def test_rejects_weights_function_with_wrong_shape(latlon: Grid, south: Grid) -> None:
    def wrong_shape(source: Locations, target: Locations) -> sparse.csr_array:
        return sparse.csr_array((source.size, target.size))

    with pytest.raises(ValueError, match="should have shape"):
        Regridder(latlon, south, wrong_shape)


@pytest.mark.parametrize(
    ("source", "target"),
    [
        (build_global_latlon(1.0), build_ease2("south", "25p0km", shape=(432, 432))),
        (
            Grid(np.arange(-60.5, 0, 1.0), np.arange(-89.5, -40, 1.0), "EPSG:4326"),
            build_ease2("south", "25p0km", shape=(432, 432)),
        ),
        (build_ease2("north", "36p0km"), build_global_latlon(1.0)),
        (
            Grid(np.arange(170.5, 200, 1.0), np.arange(60.5, 40, -1.0), "EPSG:4326"),
            build_global_latlon(2.0),
        ),
    ],
    ids=["latlon", "regional", "polar", "dateline"],
)
def test_nearest_great_circle_matches_nearest_indices(
    source: Grid, target: Grid
) -> None:
    # The indices from nearest_indices can gather data externally, e.g. torch tensors
    regridder = Regridder(source, target, "nearest_great_circle")
    src_lat, src_lon = source.latlon()
    tgt_lat, tgt_lon = target.latlon()
    iy, ix = nearest_indices(
        source_lats=src_lat,
        source_lons=src_lon,
        target_lats=tgt_lat,
        target_lons=tgt_lon,
    )
    data = np.random.default_rng(0).random((3, *source.shape))
    gathered = data[..., iy, ix]
    expected = regridder.apply(data)
    inside = ~outside(regridder)
    assert inside.any()
    np.testing.assert_array_equal(gathered[..., inside], expected[..., inside])
    assert np.isnan(expected[..., ~inside]).all()


def test_weights_are_built_once(
    latlon: Grid, south: Grid, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    real = autogrid.regrid.methods.nearest_indices
    monkeypatch.setattr(
        autogrid.regrid.methods,
        "nearest_indices",
        lambda **kwargs: calls.append(kwargs) or real(**kwargs),
    )
    regridder = Regridder(latlon, south, "nearest_great_circle")
    assert len(calls) == 1
    data = np.random.default_rng(0).random(latlon.shape)
    np.testing.assert_array_equal(regridder.apply(data), regridder.apply(data))
    assert len(calls) == 1


def test_nearest_great_circle_flags_targets_without_latlon_as_outside() -> None:
    # A north polar target grid reaching beyond the valid area of its projection, so
    # that its corner cells have no latitude or longitude
    points = np.arange(-15_000_000, 15_000_001, 300_000.0)
    target = Grid(points, points[::-1], "EPSG:6931")
    no_latlon = np.isnan(target.latlon()[0])
    assert no_latlon.any()
    regridder = Regridder(build_global_latlon(5.0), target, "nearest_great_circle")
    np.testing.assert_array_equal(outside(regridder), no_latlon)


def test_nearest_great_circle_skips_sources_without_latlon() -> None:
    # A north polar source grid reaching beyond the valid area of its projection, so
    # that its corner cells have no latitude or longitude
    points = np.arange(-15_000_000, 15_000_001, 300_000.0)
    source = Grid(points, points[::-1], "EPSG:6931")
    target = build_global_latlon(5.0)
    regridder = Regridder(source, target, "nearest_great_circle")
    data = np.random.default_rng(0).random(source.shape)
    lat, lon = target.latlon()
    expected = grid_to_points(data, grid=source, points=Points(lat, lon))
    inside = ~outside(regridder)
    assert inside.any()
    np.testing.assert_array_equal(regridder.apply(data)[inside], expected[inside])


def test_nearest_great_circle_with_no_targets_inside() -> None:
    source = Grid([0, 1, 2], [0, 1], 4326)
    target = Grid([100, 101], [50, 51], 4326)
    assert outside(Regridder(source, target, "nearest_great_circle")).all()


def test_nearest_great_circle_differs_from_nearest_near_pole(
    latlon: Grid, south: Grid
) -> None:
    flat = np.arange(latlon.size, dtype=np.float64).reshape(latlon.shape)
    differ = Regridder(latlon, south, "nearest").apply(flat) != Regridder(
        latlon, south, "nearest_great_circle"
    ).apply(flat)
    assert differ.any()
    assert south.latlon()[0][differ].min() < -89


def test_nearest_great_circle_handles_leading_dimensions_and_nan(
    latlon: Grid, south: Grid
) -> None:
    regridder = Regridder(latlon, south, "nearest_great_circle")
    data = np.random.default_rng(0).random((4, 2, *latlon.shape))
    data[..., :30, :] = np.nan  # southernmost 30 degrees
    result = regridder.apply(data)
    assert result.shape == (4, 2, *south.shape)
    np.testing.assert_array_equal(result[3, 1], regridder.apply(data[3, 1]))
    lat, _ = south.latlon()
    assert np.isnan(result[..., lat < -61]).all()
    assert not np.isnan(result[..., lat > -59]).any()


def test_nearest_great_circle_outside_source_is_nan(south: Grid) -> None:
    regional = Grid(np.arange(-60.5, 0, 1.0), np.arange(-89.5, -40, 1.0), "EPSG:4326")
    result = Regridder(regional, south, "nearest_great_circle").apply(
        np.ones(regional.shape)
    )
    assert np.isnan(result).any()
    np.testing.assert_allclose(result[~np.isnan(result)], 1.0)


def test_outside_matches_direct_transform() -> None:
    # UTM points straddling the western edge of a latitude-longitude source,
    # 20 m apart and none exactly on it
    source = Grid(np.arange(-2.0, 0.01, 0.01), np.arange(52.0, 53.01, 0.01), 4326)
    edge = source.x_bounds[0, 0]
    x0, y0 = Transformer.from_crs(4326, 32630, always_xy=True).transform(edge, 52.5)
    target = Grid(x0 + np.arange(-490, 500, 20.0), [y0, y0 + 20], 32630)
    # Transform straight from the target CRS, not through latitude and longitude
    x, y = Transformer.from_crs(32630, 4326, always_xy=True).transform(
        *np.meshgrid(target.x_points, target.y_points)
    )
    want = ~source.contains(y=y, x=x)
    np.testing.assert_array_equal(outside(Regridder(source, target, "linear")), want)
    assert want.any()
    assert not want.all()
    # nearest_great_circle leaves the same points NaN
    regridder = Regridder(source, target, "nearest_great_circle")
    np.testing.assert_array_equal(outside(regridder), want)


def test_every_method_is_registered() -> None:
    assert set(METHODS) == set(get_args(Method))


@pytest.mark.parametrize("method", get_args(Method))
def test_outside_source_is_nan_for_each_method(method: Method) -> None:
    # A lat-lon target, as area weighting requires the source coordinate system
    regional = Grid(np.arange(-60.5, 0, 1.0), np.arange(-89.5, -40, 1.0), "EPSG:4326")
    nan = outside(Regridder(regional, build_global_latlon(2.0), method))
    assert nan.any()
    assert not nan.all()


def test_area_weighted_nan_spreads_to_overlapping_cells(latlon: Grid) -> None:
    data = np.ones(latlon.shape)
    data[100, 200] = np.nan
    regridder = Regridder(latlon, build_global_latlon(2.0), "area_weighted")
    result = regridder.apply(data)
    assert np.isnan(result).sum() == 1
    assert np.isnan(result[50, 100])


def test_area_weighted_wraps_longitude() -> None:
    source = build_global_latlon(1.0)
    data = np.broadcast_to(np.arange(source.shape[1], dtype=np.float64), source.shape)
    # Cells offset by half a degree, the first straddling the source's first
    # and last columns
    target = Grid(np.arange(-180.0, 180.0, 1.0), [0.0, 10.0], "EPSG:4326")
    result = Regridder(source, target, "area_weighted").apply(data)
    np.testing.assert_allclose(result[:, 0], (data[0, 0] + data[0, -1]) / 2)
    np.testing.assert_allclose(result[:, 1:], data[:2, :-1] + 0.5)


def test_area_weighted_rejects_different_crs(latlon: Grid, south: Grid) -> None:
    with pytest.raises(ValueError, match="share a CRS"):
        Regridder(latlon, south, "area_weighted")


def _random_points(n: int, seed: int = 0) -> Points:
    rng = np.random.default_rng(seed)
    return Points(
        np.rad2deg(np.arcsin(rng.uniform(-1, 1, n))), rng.uniform(-180, 180, n)
    )


def test_grid_is_locations(latlon: Grid) -> None:
    locations: Locations = latlon
    assert locations.shape == latlon.shape


def test_grid_to_points_regridder_matches_grid_to_points(latlon: Grid) -> None:
    targets = _random_points(50)
    regridder = Regridder(latlon, targets, "nearest_great_circle")
    data = np.random.default_rng(1).random((3, *latlon.shape))
    lats, lons = targets.latlon()
    expected = grid_to_points(data, grid=latlon, points=Points(lats, lons))
    result = regridder.apply(data)
    assert result.shape == (3, 50)
    np.testing.assert_array_equal(result, expected)


def test_grid_to_points_regridder_masks_points_outside_grid() -> None:
    regional = Grid(np.arange(-60.5, 0, 1.0), np.arange(-89.5, -40, 1.0), "EPSG:4326")
    targets = Points([-60.0, 10.0], [-30.0, 100.0])
    result = Regridder(regional, targets, "nearest_great_circle").apply(
        np.ones(regional.shape)
    )
    np.testing.assert_array_equal(result, [1.0, np.nan])


@pytest.mark.parametrize("method", ["linear", "nearest"])
def test_grid_to_points_matches_grid_to_grid(
    latlon: Grid, south: Grid, method: Method
) -> None:
    # Regridding to the points of a grid gives the same values, flattened
    lats, lons = south.latlon()
    data = analytic(latlon)
    result = Regridder(latlon, Points(lats.ravel(), lons.ravel()), method).apply(data)
    expected = Regridder(latlon, south, method).apply(data)
    np.testing.assert_array_equal(result, expected.ravel())


def test_points_to_points_matches_nearest_indices() -> None:
    source, target = _random_points(500), _random_points(250, seed=1)
    data = np.random.default_rng(2).random((5, 500))
    source_lats, source_lons = source.latlon()
    target_lats, target_lons = target.latlon()
    (index,) = nearest_indices(
        source_lats=source_lats,
        source_lons=source_lons,
        target_lats=target_lats,
        target_lons=target_lons,
    )
    result = Regridder(source, target, "nearest_great_circle").apply(data)
    # Points have no cells, so no target points are outside
    np.testing.assert_array_equal(result, data[..., index])


def test_curvilinear_points_regrid_to_grid(south: Grid) -> None:
    # A 2D source with no rectilinear structure, as for a curvilinear grid
    lats, lons = south.latlon()
    source = Points(lats, lons)
    target = build_global_latlon(5.0)
    data = np.random.default_rng(0).random((2, *south.shape))
    result = Regridder(source, target, "nearest_great_circle").apply(data)
    assert result.shape == (2, *target.shape)
    assert not np.isnan(result).any()
    # The same nearest points as from the grid, inside it
    from_grid = Regridder(south, target, "nearest_great_circle").apply(data)
    inside = ~np.isnan(from_grid)
    np.testing.assert_array_equal(result[inside], from_grid[inside])


def test_points_to_grid_leaves_targets_without_latlon_nan() -> None:
    points = np.arange(-15_000_000, 15_000_001, 300_000.0)
    target = Grid(points, points[::-1], "EPSG:6931")
    no_latlon = np.isnan(target.latlon()[0])
    regridder = Regridder(_random_points(100), target, "nearest_great_circle")
    np.testing.assert_array_equal(outside(regridder), no_latlon)


@pytest.mark.parametrize("method", ["linear", "nearest", "area_weighted"])
def test_grid_methods_reject_points_source(latlon: Grid, method: Method) -> None:
    with pytest.raises(TypeError, match="needs a Grid source"):
        Regridder(_random_points(10), latlon, method)


def test_area_weighted_rejects_points_target(latlon: Grid) -> None:
    with pytest.raises(TypeError, match="needs a Grid target"):
        Regridder(latlon, _random_points(10), "area_weighted")


def test_points_source_rejects_wrong_shape(latlon: Grid) -> None:
    regridder = Regridder(_random_points(10), latlon, "nearest_great_circle")
    with pytest.raises(ValueError, match="does not end with the source shape"):
        regridder.apply(np.ones((10, 3)))
