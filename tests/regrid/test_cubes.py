import numpy as np
import pytest

from autogrid import Grid
from autogrid.regrid.cubes import to_cube


@pytest.mark.parametrize("crs", ["EPSG:6931", "EPSG:32630"])
def test_projected_coords_are_in_metres(crs: str) -> None:
    cube = to_cube(Grid(np.arange(3.0), np.arange(4.0), crs))
    for name in ("projection_x_coordinate", "projection_y_coordinate"):
        assert cube.coord(name).units == "m"
