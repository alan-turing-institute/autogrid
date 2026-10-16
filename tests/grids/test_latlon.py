import numpy as np
import pytest

from autogrid import Grid
from autogrid.grids import build_global_latlon


def test_latlon_grid() -> None:
    grid = build_global_latlon(1.0)
    assert grid.shape == (180, 360)
    assert grid.x_points[0] == -179.5
    assert grid.y_points[-1] == 89.5
    assert grid.is_circular


def test_latlon_grid_rejects_uneven_resolution() -> None:
    with pytest.raises(ValueError, match="does not divide"):
        build_global_latlon(0.7)


@pytest.mark.parametrize("resolution", [0, -1.0, -180, 360, np.inf, np.nan])
def test_latlon_grid_rejects_out_of_range_resolution(resolution: float) -> None:
    with pytest.raises(ValueError, match="must be in"):
        build_global_latlon(resolution)


def test_latlon_grid_equals_plain_grid() -> None:
    lon = np.arange(-179.5, 180, 1.0)
    lat = np.arange(-89.5, 90, 1.0)
    assert build_global_latlon(1.0) == Grid(lon, lat, "EPSG:4326")
