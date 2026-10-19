import cartopy.crs as ccrs
import numpy as np
import pytest
from pyproj import CRS

from autogrid import Grid
from autogrid.grids import build_ease2
from autogrid.grids.ease2 import Region


@pytest.mark.parametrize("region", ["north", "south", "global"])
def test_ease2_projection_has_plot_limits(region: Region) -> None:
    x_min, x_max = build_ease2(region, "36p0km").projection.x_limits
    assert x_max - x_min > 1e7


def test_rotated_grid_projection_is_rotated_pole() -> None:
    crs = CRS.from_cf(
        {
            "grid_mapping_name": "rotated_latitude_longitude",
            "grid_north_pole_latitude": 37.5,
            "grid_north_pole_longitude": 177.5,
        }
    )
    projection = Grid(np.linspace(-5, 5, 20), np.linspace(-5, 5, 30), crs).projection
    assert isinstance(projection, ccrs.RotatedPole)
    assert projection.proj4_params["o_lat_p"] == 37.5


@pytest.mark.parametrize("crs", ["EPSG:4326", "OGC:CRS84", "+proj=longlat +R=6371229"])
def test_plain_latlon_projection_is_plate_carree(crs: str) -> None:
    assert isinstance(Grid([0.0, 1.0], [0.0, 1.0], crs).projection, ccrs.PlateCarree)


def test_sphere_projection_keeps_radius() -> None:
    globe = Grid([0.0, 1.0], [0.0, 1.0], "+proj=longlat +R=6371229").projection.globe
    assert globe is not None
    assert globe.semimajor_axis == globe.semiminor_axis == 6371229
