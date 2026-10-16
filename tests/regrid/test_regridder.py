import numpy as np
import pytest
from pyproj import Transformer

import autogrid.regrid.nearest
from autogrid import Grid, Regridder
from autogrid.grids import build_global_latlon
from autogrid.regrid.nearest import indices_grid_to_grid, indices_points_to_points
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
    # The indices from nearest can gather data externally, e.g. torch tensors
    regridder = Regridder(source, south)
    iy, ix = indices_grid_to_grid(source, south)
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
    real = autogrid.regrid.nearest.indices_grid_to_grid
    monkeypatch.setattr(
        autogrid.regrid.nearest,
        "indices_grid_to_grid",
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
    got = indices_grid_to_grid(grid, target)
    expected = indices_points_to_points(
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


def test_global_grads_grid_is_circular() -> None:
    grid = Grid(np.arange(0, 400, 10.0) + 5, np.arange(-95, 100, 10.0), "EPSG:4807")
    assert grid.is_circular
