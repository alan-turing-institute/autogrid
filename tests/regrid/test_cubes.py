import numpy as np
import pytest

from autogrid import Grid
from autogrid.grids import build_global_latlon
from autogrid.regrid.cubes import pad_edges, padded_coords, to_cube


@pytest.mark.parametrize("crs", ["EPSG:6931", "EPSG:32630"])
def test_projected_coords_are_in_metres(crs: str) -> None:
    cube = to_cube(Grid(np.arange(3.0), np.arange(4.0), crs))
    for name in ("projection_x_coordinate", "projection_y_coordinate"):
        assert cube.coord(name).units == "m"


def test_padded_coords_add_a_point_at_each_cell_edge() -> None:
    grid = Grid(np.arange(-10.0, 10.0, 1.0), np.arange(50.0, 60.0, 0.5), "EPSG:4326")
    (y, x), pad = padded_coords(grid)
    assert pad == ((1, 1), (1, 1))
    np.testing.assert_array_equal(x.points[[0, -1]], grid.x_bounds[[0, -1], [0, 1]])
    np.testing.assert_array_equal(y.points[[0, -1]], grid.y_bounds[[0, -1], [0, 1]])
    np.testing.assert_array_equal(x.points[1:-1], grid.x_points)
    assert x.bounds is None
    assert y.bounds is None


def test_padded_coords_do_not_pad_circular_x_or_beyond_the_poles() -> None:
    grid = Grid(np.arange(0.0, 360.0, 1.0), np.arange(-90.0, 90.5, 1.0), "EPSG:4326")
    (y, x), pad = padded_coords(grid)
    assert pad == ((0, 0), (0, 0))
    np.testing.assert_array_equal(x.points, grid.x_points)
    np.testing.assert_array_equal(y.points, grid.y_points)
    (y, _), pad = padded_coords(build_global_latlon(1.0))
    assert pad == ((1, 1), (0, 0))
    np.testing.assert_array_equal(y.points[[0, -1]], [-90.0, 90.0])


def test_pad_edges_repeats_values_and_mask() -> None:
    data = np.ma.masked_invalid([[[1.0, np.nan], [3.0, 4.0]]])
    padded = pad_edges(data, ((1, 0), (0, 1)))
    assert padded.shape == (1, 3, 3)
    np.testing.assert_array_equal(padded.data[0, 0], padded.data[0, 1])
    np.testing.assert_array_equal(padded.data[0, 2], [3.0, 4.0, 4.0])
    np.testing.assert_array_equal(
        np.ma.getmaskarray(padded)[0, :, 1:], [[1, 1], [1, 1], [0, 0]]
    )
