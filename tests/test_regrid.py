import numpy as np
import pytest
from pyproj import Transformer

from autogrid import Grid, Regridder
from autogrid.grids import build_global_latlon
from autogrid.nearest import nearest_indices
from autogrid.regridder import Method, NearestMethod


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


@pytest.mark.parametrize("method", ["nearest", "nearest_great_circle"])
@pytest.mark.parametrize(
    "source",
    [
        build_global_latlon(1.0),
        Grid(np.arange(-60.5, 0, 1.0), np.arange(-89.5, -40, 1.0), "EPSG:4326"),
    ],
    ids=["global", "regional"],
)
def test_nearest_indices_match_apply(
    source: Grid, south: Grid, method: NearestMethod
) -> None:
    regridder = Regridder(source, south)
    iy, ix = regridder.nearest_indices(method)
    assert iy.shape == ix.shape == south.shape
    assert iy.dtype == ix.dtype == np.intp
    assert iy.min() >= 0 and iy.max() < source.shape[0]
    assert ix.min() >= 0 and ix.max() < source.shape[1]
    data = np.random.default_rng(0).random((3, *source.shape))
    gathered = data[..., iy, ix]
    expected = regridder.apply(data, method)
    inside = ~regridder.outside_source
    np.testing.assert_array_equal(gathered[..., inside], expected[..., inside])
    assert np.isnan(expected[..., ~inside]).all()


def test_nearest_indices_are_cached_per_method(latlon: Grid, south: Grid) -> None:
    regridder = Regridder(latlon, south)
    nearest = regridder.nearest_indices("nearest")
    great_circle = regridder.nearest_indices("nearest_great_circle")
    assert regridder.nearest_indices("nearest") is nearest
    assert regridder.nearest_indices("nearest_great_circle") is great_circle
    assert not all(np.array_equal(a, b) for a, b in zip(nearest, great_circle))


def test_nearest_indices_rejects_other_methods(latlon: Grid, south: Grid) -> None:
    with pytest.raises(ValueError, match="Unknown nearest neighbour method"):
        Regridder(latlon, south).nearest_indices("linear")  # ty: ignore[invalid-argument-type]


def _fake_nearest(regridder: Regridder, invalid: np.ndarray, masked: bool) -> None:
    """Make the cached "nearest" regridder return flat indices with gaps."""
    real = regridder._regridder("nearest")

    def fake(cube):
        result = real(cube)
        data = np.ma.masked_invalid(result.data)
        if masked:
            data[invalid] = np.ma.masked
        else:
            data = np.where(invalid, np.nan, data)
        result.data = data
        return result

    regridder._regridders["nearest"] = fake


@pytest.mark.parametrize("masked", [False, True], ids=["nan", "masked"])
def test_nearest_indices_invalid_outside_source(south: Grid, masked: bool) -> None:
    regional = Grid(np.arange(-60.5, 0, 1.0), np.arange(-89.5, -40, 1.0), "EPSG:4326")
    regridder = Regridder(regional, south)
    _fake_nearest(regridder, regridder.outside_source, masked)
    iy, ix = regridder.nearest_indices("nearest")
    assert iy.dtype == ix.dtype == np.intp
    assert iy.min() >= 0 and iy.max() < regional.shape[0]
    assert ix.min() >= 0 and ix.max() < regional.shape[1]


@pytest.mark.parametrize("masked", [False, True], ids=["nan", "masked"])
def test_nearest_indices_invalid_inside_source_raises(
    latlon: Grid, south: Grid, masked: bool
) -> None:
    regridder = Regridder(latlon, south)
    invalid = np.zeros(south.shape, dtype=bool)
    invalid[200, 200] = True
    _fake_nearest(regridder, invalid, masked)
    with pytest.raises(RuntimeError, match="no valid source index for 1 target"):
        regridder.nearest_indices("nearest")


@pytest.mark.parametrize("method", ["nearest", "nearest_great_circle"])
def test_nearest_indices_gather_torch_tensors(
    latlon: Grid, south: Grid, method: NearestMethod
) -> None:
    torch = pytest.importorskip("torch")
    regridder = Regridder(latlon, south)
    iy, ix = (torch.from_numpy(i) for i in regridder.nearest_indices(method))
    data = np.random.default_rng(0).random((2, 4, *latlon.shape)).astype(np.float32)
    gathered = torch.from_numpy(data)[:, :, iy, ix]
    np.testing.assert_array_equal(gathered.numpy(), regridder.apply(data, method))


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
    got = Regridder(grid, target).nearest_indices("nearest_great_circle")
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


def test_regrid_rejects_latlon_not_in_degrees() -> None:
    source = Grid(np.arange(0, 10.0), np.arange(40, 50.0), "EPSG:4807")
    with pytest.raises(ValueError, match="must be in degrees"):
        Regridder(source, build_global_latlon(1.0)).apply(np.zeros(source.shape))
