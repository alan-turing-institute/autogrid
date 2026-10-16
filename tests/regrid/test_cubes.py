import numpy as np
import pytest
from cf_units import Unit

from autogrid import Grid
from autogrid.regrid.cubes import to_cube


@pytest.mark.parametrize(
    ("crs", "metres"),
    [
        ("EPSG:27700", 1.0),
        ("EPSG:2263", 0.3048006096012192),
        ("+proj=utm +zone=30 +units=km", 1000.0),
        ("+proj=utm +zone=30 +units=ind-ft", 0.30479841),
    ],
)
def test_projected_coords_have_crs_units(crs: str, metres: float) -> None:
    cube = to_cube(Grid(np.arange(3.0), np.arange(4.0), crs))
    for name in ("projection_x_coordinate", "projection_y_coordinate"):
        units = cube.coord(name).units
        assert isinstance(units, Unit)
        np.testing.assert_allclose(units.convert(1, "m"), metres)
