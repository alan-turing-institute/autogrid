import pytest
from iris.coord_systems import GeogCS, RotatedGeogCS
from pyproj import CRS

from autogrid.crs.iris import coord_system


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
ROTATED_SPHERE = CRS.from_cf(
    {
        "grid_mapping_name": "rotated_latitude_longitude",
        "grid_north_pole_latitude": 37.5,
        "grid_north_pole_longitude": 177.5,
        "earth_radius": 6371229.0,
    }
)


@pytest.mark.parametrize("crs", [ROTATED, ROTATED_SPHERE], ids=["wgs84", "sphere"])
def test_rotated_pole_is_rotated_geog_cs(crs: CRS) -> None:
    cs = coord_system(crs)
    assert isinstance(cs, RotatedGeogCS)
    assert cs.grid_north_pole_latitude == 37.5
    assert cs.grid_north_pole_longitude == 177.5


def test_sphere_keeps_radius() -> None:
    cs = coord_system(CRS("+proj=longlat +R=6371229"))
    assert isinstance(cs, GeogCS)
    assert cs.semi_major_axis == cs.semi_minor_axis == 6371229
