import numpy as np
import pytest

from autogrid.crs.cartopy import projection
from autogrid.grids import build_ease2
from autogrid.grids.ease2 import Region


@pytest.mark.parametrize(
    ("region", "epsg"), [("north", 6931), ("south", 6932), ("global", 6933)]
)
def test_ease2_crs(region: Region, epsg: int) -> None:
    grid = build_ease2(region, "36p0km")
    assert grid.crs.to_epsg() == epsg


def test_ease2_south_25p0km() -> None:
    grid = build_ease2("south", "25p0km")
    assert grid.shape == (720, 720)
    assert grid.crs.to_epsg() == 6932
    # Rows run top to bottom, cell centres half a cell in from the edge
    assert grid.x_points[0] == -8_987_500
    assert grid.y_points[0] == 8_987_500
    assert np.isclose(np.min(grid.latitudes), -90, atol=0.2)


def test_ease2_global_extent() -> None:
    grid = build_ease2("global", "36p0km")
    # The global grid spans the whole width of the projection
    x_min, x_max = projection(grid).x_limits
    assert np.isclose(grid.x_bounds[0, 0], x_min)
    assert np.isclose(grid.x_bounds[-1, 1], x_max)


def test_ease2_subset_aligns_with_full_grid() -> None:
    full = build_ease2("south", "25p0km")
    subset = build_ease2("south", "25p0km", shape=(432, 432))
    offset = (720 - 432) // 2
    np.testing.assert_array_equal(subset.x_points, full.x_points[offset : offset + 432])


@pytest.mark.parametrize("shape", [(431, 431), (800, 800), (0, 0), (-2, 432)])
def test_ease2_rejects_misaligned_subset(shape: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="centred subset"):
        build_ease2("south", "25p0km", shape=shape)


@pytest.mark.parametrize(
    ("region", "resolution", "match"),
    [
        ("east", "25p0km", "Unknown region"),
        ("south", "25km", "Unknown resolution"),
        ("global", "10p0km", "Unknown resolution"),
    ],
)
def test_ease2_rejects_unknown_spec(region: str, resolution: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        build_ease2(region, resolution)  # ty: ignore[invalid-argument-type]
