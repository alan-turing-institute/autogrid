import numpy as np
import pytest
from cf_units import Unit
from iris.coord_systems import GeogCS, RotatedGeogCS
from pyproj import CRS

from autogrid import Grid
from autogrid.regrid.cubes import CartopyRotatedGeogCS, coord_system, to_cube


def test_coord_systems_are_not_shared() -> None:
    crs = CRS("EPSG:4326")
    first, second = coord_system(crs), coord_system(crs)
    assert isinstance(first, GeogCS)
    assert isinstance(second, GeogCS)
    first.semi_major_axis = 1.0
    assert second.semi_major_axis != 1.0


ROTATED = CRS.from_cf(
    {
        "grid_mapping_name": "rotated_latitude_longitude",
        "grid_north_pole_latitude": 37.5,
        "grid_north_pole_longitude": 177.5,
    }
)
# The OSGB 1936 datum shift is more than its ellipsoid can describe
ROTATED_OSGB = CRS(
    "+proj=ob_tran +o_proj=longlat +o_lon_p=0 +o_lat_p=37.5 +lon_0=357.5 "
    "+ellps=airy +towgs84=446.448,-125.157,542.06,0.15,0.247,0.842,-20.489"
)


@pytest.mark.parametrize(
    ("crs", "cs_type"),
    [(ROTATED, RotatedGeogCS), (ROTATED_OSGB, CartopyRotatedGeogCS)],
    ids=["globe", "datum-shift"],
)
def test_rotated_pole_is_rotated_geog_cs(crs: CRS, cs_type: type) -> None:
    cs = coord_system(crs)
    assert isinstance(cs, RotatedGeogCS)
    assert type(cs) is cs_type
    assert cs.grid_north_pole_latitude == 37.5
    assert cs.grid_north_pole_longitude == 177.5


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
