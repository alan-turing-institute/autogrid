import numpy as np
import pytest
from iris.analysis._regrid import RectilinearRegridder
from pyproj import CRS, Transformer

import autogrid.regrid.grid_search
from autogrid import Grid, Regridder
from autogrid.grids import build_global_latlon
from autogrid.regrid.cubes import to_cube
from autogrid.regrid.grid_search import grid_to_grid
from autogrid.regrid.points import nearest_indices
from autogrid.regrid.regridder import Method


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
    np.testing.assert_allclose(result, analytic(south), atol=atol)


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
    linear = regridder._regridders["linear"]
    regridder.apply(data, "linear")
    regridder.apply(data, "nearest")
    assert set(regridder._regridders) == {"linear", "nearest"}
    assert regridder._regridders["linear"] is linear


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
    inside = ~regridder.outside_source
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


def test_nearest_great_circle_uses_source_datum() -> None:
    # On British National Grid (OSGB36 datum) the nearest points must be found
    # by the same latitudes and longitudes as Grid, with no datum shift
    grid = Grid(np.arange(0, 700000, 50000.0), np.arange(0, 1200000, 50000.0), 27700)
    # A dense target, so a datum shift of ~100 m would move some nearest points
    target = Grid(np.arange(-8, 2, 0.05), np.arange(50, 60, 0.05), "EPSG:4326")
    got = grid_to_grid(grid, target)
    expected = nearest_indices(
        grid.latitudes, grid.longitudes, target.latitudes, target.longitudes
    )
    for g, want in zip(got, expected, strict=True):
        np.testing.assert_array_equal(g, want)


@pytest.mark.parametrize("method", ["linear", "nearest", "nearest_great_circle"])
def test_regrid_respects_prime_meridian(method: Method) -> None:
    # Longitudes relative to the Paris meridian, 2.34 degrees east of Greenwich
    source = Grid(
        np.arange(-10, 10.01, 0.1),
        np.arange(40, 52.01, 0.1),
        "+proj=longlat +ellps=clrk80ign +pm=paris +no_defs",
    )
    target = Grid(np.arange(-5, 10.01, 0.5), np.arange(42, 50.01, 0.5), "EPSG:4326")
    paris_degrees = 2.5969213 * 0.9
    result = Regridder(source, target).apply(source.x + paris_degrees, method)
    np.testing.assert_allclose(result, target.x, atol=0.05)


def test_regrid_respects_datum() -> None:
    # OSGB 1936 is offset from WGS 84 by about 0.002 degrees of longitude here
    source = Grid(np.arange(-5, 1.01, 0.05), np.arange(50, 56.01, 0.05), "EPSG:4277")
    target = Grid(np.arange(-4, 0.01, 0.5), np.arange(51, 55.01, 0.5), "EPSG:4326")
    result = Regridder(source, target).apply(source.x, "linear")
    expected, _ = Transformer.from_crs(
        target.crs, source.crs, always_xy=True
    ).transform(target.x, target.y)
    np.testing.assert_allclose(result, expected, atol=1e-5)


def test_outside_source_matches_iris_sample_points() -> None:
    # British National Grid points (OSGB36 datum) straddling the western edge
    # of a WGS 84 source, 20 m apart, so the ~100 m datum shift that iris
    # applies moves the edge by several points
    source = Grid(np.arange(-2.0, 0.01, 0.01), np.arange(52.0, 53.01, 0.01), 4326)
    edge = source.x_bounds[0, 0]
    x0, y0 = Transformer.from_crs(4326, 27700, always_xy=True).transform(edge, 52.5)
    target = Grid(x0 + np.arange(-500, 500, 20.0), [y0, y0 + 20], 27700)
    regridder = Regridder(source, target)
    src_y, _ = to_cube(source).dim_coords
    tgt_y, tgt_x = to_cube(target).dim_coords
    x, y = RectilinearRegridder._sample_grid(src_y.coord_system, tgt_x, tgt_y)
    want = ~source.contains(x, y)
    np.testing.assert_array_equal(regridder.outside_source, want)
    assert want.any()
    assert not want.all()
    # The great circle search compares latitudes and longitudes unshifted
    _, _, unshifted = regridder._great_circle
    assert (unshifted != want).any()
    nan = np.isnan(regridder.apply(np.ones(source.shape), "nearest_great_circle"))
    np.testing.assert_array_equal(nan, unshifted)


def test_regrid_rejects_latlon_not_in_degrees() -> None:
    source = Grid(np.arange(0, 10.0), np.arange(40, 50.0), "EPSG:4807")
    with pytest.raises(ValueError, match="must be in degrees"):
        Regridder(source, build_global_latlon(1.0)).apply(np.zeros(source.shape))


def test_outside_source_handles_grads() -> None:
    # EPSG:4807 is in grads relative to Paris, so a full turn of longitude is 400
    # The source spans 45 to 355 grads, so -50 grads (350) is inside it but
    # -30 grads (370) is not
    source = Grid(np.arange(50, 351, 10.0), np.arange(-50, 51, 10.0), "EPSG:4807")
    target = Grid([-50.0, -30.0], [0.0, 10.0], "EPSG:4807")
    outside = Regridder(source, target).outside_source
    np.testing.assert_array_equal(outside, [[False, True], [False, True]])


def test_outside_source_handles_grads_from_degrees() -> None:
    # Iris can't build a coordinate system for EPSG:4807, as it is in grads, but
    # outside_source needs none. The source spans 45 to 355 grads east of Paris,
    # which is 2.34 degrees east, so from 42.84 degrees east, eastward to 38.16
    # degrees west
    source = Grid(np.arange(50, 351, 10.0), np.arange(-50, 51, 10.0), "EPSG:4807")
    target = Grid([-40.0, -36.0], [0.0, 10.0], "EPSG:4326")
    outside = Regridder(source, target).outside_source
    np.testing.assert_array_equal(outside, [[False, True], [False, True]])


def test_global_grads_grid_is_circular() -> None:
    grid = Grid(np.arange(0, 400, 10.0) + 5, np.arange(-95, 100, 10.0), "EPSG:4807")
    assert grid.is_circular
