import numpy as np
import pytest
from iris.coords import DimCoord

from autogrid import Grid, Regridder, nearest_indices
from autogrid.adaptors.iris import NearestGreatCircle, to_cube
from autogrid.grids import EASE2Grid, LatLonGrid


def analytic(grid: Grid) -> np.ndarray:
    return np.cos(np.deg2rad(grid.latitudes)) * np.sin(np.deg2rad(grid.longitudes))


@pytest.mark.parametrize(
    ("method", "atol"),
    [("linear", 1e-3), ("nearest", 2e-2), ("nearest_great_circle", 2e-2)],
)
def test_regrid_matches_analytic_field(
    latlon: LatLonGrid, south: EASE2Grid, method: str, atol: float
) -> None:
    result = Regridder(latlon, south).apply(analytic(latlon).astype(np.float32), method)
    assert isinstance(result, np.ndarray)
    assert not isinstance(result, np.ma.MaskedArray)
    assert result.shape == south.shape
    np.testing.assert_allclose(result, analytic(south), atol=atol)


def test_area_weighted_conserves_mean(latlon: LatLonGrid) -> None:
    data = np.random.default_rng(0).random(latlon.shape)
    result = Regridder(latlon, LatLonGrid(2.0)).apply(data, "area_weighted")
    assert result.shape == (90, 180)
    weights = np.cos(np.deg2rad(latlon.latitudes))
    coarse_weights = np.cos(np.deg2rad(LatLonGrid(2.0).latitudes))
    np.testing.assert_allclose(
        np.average(result, weights=coarse_weights),
        np.average(data, weights=weights),
        rtol=1e-3,
    )


def test_regridder_handles_leading_dimensions(
    latlon: LatLonGrid, south: EASE2Grid
) -> None:
    regridder = Regridder(latlon, south)
    data = np.random.default_rng(0).random((4, 2, *latlon.shape)).astype(np.float32)
    result = regridder.apply(data)
    assert result.shape == (4, 2, *south.shape)
    np.testing.assert_array_equal(result[3, 1], regridder.apply(data[3, 1]))


def test_points_outside_source_are_nan(south: EASE2Grid) -> None:
    regional = Grid(np.arange(-60.5, 0, 1.0), np.arange(-89.5, -40, 1.0), "EPSG:4326")
    result = Regridder(regional, south).apply(np.ones(regional.shape))
    assert np.isnan(result).any()
    np.testing.assert_allclose(result[~np.isnan(result)], 1.0)


def test_nan_input_propagates(latlon: LatLonGrid, south: EASE2Grid) -> None:
    data = np.ones(latlon.shape)
    data[:30] = np.nan  # southernmost 30 degrees
    result = Regridder(latlon, south).apply(data)
    assert np.isnan(result[south.latitudes < -61]).all()
    assert not np.isnan(result[south.latitudes > -59]).any()


def test_rejects_wrong_shape(latlon: LatLonGrid, south: EASE2Grid) -> None:
    with pytest.raises(ValueError, match="does not end with the source grid shape"):
        Regridder(latlon, south).apply(np.ones((10, 10)))


def test_rejects_unknown_method(latlon: LatLonGrid, south: EASE2Grid) -> None:
    with pytest.raises(ValueError, match="Unknown regridding method"):
        Regridder(latlon, south).apply(np.ones(latlon.shape), "cubic")


def test_regridder_caches_one_regridder_per_method(
    latlon: LatLonGrid, south: EASE2Grid
) -> None:
    regridder = Regridder(latlon, south)
    data = np.ones(latlon.shape)
    regridder.apply(data, "linear")
    linear = regridder._regridders["linear"]
    regridder.apply(data, "linear")
    regridder.apply(data, "nearest")
    assert set(regridder._regridders) == {"linear", "nearest"}
    assert regridder._regridders["linear"] is linear


def test_methods_differ_on_same_regridder(latlon: LatLonGrid, south: EASE2Grid) -> None:
    regridder = Regridder(latlon, south)
    data = analytic(latlon)
    assert not np.array_equal(
        regridder.apply(data, "linear"), regridder.apply(data, "nearest")
    )


@pytest.mark.parametrize("method", ["nearest", "nearest_great_circle"])
@pytest.mark.parametrize(
    "source",
    [
        LatLonGrid(1.0),
        Grid(np.arange(-60.5, 0, 1.0), np.arange(-89.5, -40, 1.0), "EPSG:4326"),
    ],
    ids=["global", "regional"],
)
def test_nearest_indices_match_apply(
    source: Grid, south: EASE2Grid, method: str
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


def test_nearest_indices_are_cached_per_method(
    latlon: LatLonGrid, south: EASE2Grid
) -> None:
    regridder = Regridder(latlon, south)
    nearest = regridder.nearest_indices("nearest")
    great_circle = regridder.nearest_indices("nearest_great_circle")
    assert regridder.nearest_indices("nearest") is nearest
    assert regridder.nearest_indices("nearest_great_circle") is great_circle
    assert not all(np.array_equal(a, b) for a, b in zip(nearest, great_circle))


def test_nearest_indices_rejects_other_methods(
    latlon: LatLonGrid, south: EASE2Grid
) -> None:
    with pytest.raises(ValueError, match="Unknown nearest neighbour method"):
        Regridder(latlon, south).nearest_indices("linear")


@pytest.mark.parametrize("method", ["nearest", "nearest_great_circle"])
def test_nearest_indices_gather_torch_tensors(
    latlon: LatLonGrid, south: EASE2Grid, method: str
) -> None:
    torch = pytest.importorskip("torch")
    regridder = Regridder(latlon, south)
    iy, ix = (torch.from_numpy(i) for i in regridder.nearest_indices(method))
    data = np.random.default_rng(0).random((2, 4, *latlon.shape)).astype(np.float32)
    gathered = torch.from_numpy(data)[:, :, iy, ix]
    np.testing.assert_array_equal(gathered.numpy(), regridder.apply(data, method))


def test_nearest_great_circle_differs_from_nearest_near_pole(
    latlon: LatLonGrid, south: EASE2Grid
) -> None:
    regridder = Regridder(latlon, south)
    flat = np.arange(latlon.x.size, dtype=np.float64).reshape(latlon.shape)
    differ = regridder.apply(flat, "nearest") != regridder.apply(
        flat, "nearest_great_circle"
    )
    assert differ.any()
    assert south.latitudes[differ].min() < -89


def test_nearest_great_circle_handles_leading_dimensions_and_nan(
    latlon: LatLonGrid, south: EASE2Grid
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


def test_nearest_great_circle_outside_source_is_nan(south: EASE2Grid) -> None:
    regional = Grid(np.arange(-60.5, 0, 1.0), np.arange(-89.5, -40, 1.0), "EPSG:4326")
    result = Regridder(regional, south).apply(
        np.ones(regional.shape), "nearest_great_circle"
    )
    assert np.isnan(result).any()
    np.testing.assert_allclose(result[~np.isnan(result)], 1.0)


def test_nearest_great_circle_works_as_iris_scheme(
    latlon: LatLonGrid, south: EASE2Grid
) -> None:
    source = to_cube(latlon, np.ones((3, *latlon.shape)))
    source.long_name = "sea_ice"
    source.add_dim_coord(
        DimCoord([0, 1, 2], long_name="time", units="days since 2020-01-01"), 0
    )
    target = to_cube(south)
    result = source.regrid(target, NearestGreatCircle())
    assert result.shape == (3, *south.shape)
    assert result.long_name == "sea_ice"
    assert result.coord("time") == source.coord("time")
    assert result.coord(axis="x") == target.coord(axis="x")


def test_nearest_great_circle_rejects_cube_on_other_grid(
    latlon: LatLonGrid, south: EASE2Grid
) -> None:
    regridder = NearestGreatCircle().regridder(to_cube(latlon), to_cube(south))
    with pytest.raises(ValueError, match="not on the source grid"):
        regridder(to_cube(LatLonGrid(2.0), np.zeros((90, 180))))


def test_nearest_great_circle_uses_source_datum() -> None:
    # On British National Grid (OSGB36 datum) the scheme must find the nearest
    # points by the same latitudes and longitudes as Grid, with no datum shift
    grid = Grid(np.arange(0, 700000, 50000.0), np.arange(0, 1200000, 50000.0), 27700)
    # A dense target, so a datum shift of ~100 m would move some nearest points
    target = Grid(np.arange(-8, 2, 0.05), np.arange(50, 60, 0.05), "EPSG:4326")
    regridder = NearestGreatCircle().regridder(to_cube(grid), to_cube(target))
    expected = nearest_indices(
        grid.latitudes, grid.longitudes, target.latitudes, target.longitudes
    )
    for got, want in zip(regridder.indices, expected, strict=True):
        np.testing.assert_array_equal(got, want)
