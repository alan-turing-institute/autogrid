import cartopy.crs as ccrs
import numpy as np
import pytest
from pyproj import CRS

from autogrid import Grid
from autogrid.grids import build_ease2
from autogrid.grids.ease2 import Region
from autogrid.plot import projection


@pytest.mark.parametrize("region", ["north", "south", "global"])
def test_ease2_projection_has_plot_limits(region: Region) -> None:
    x_min, x_max = projection(build_ease2(region, "36p0km")).x_limits
    assert x_max - x_min > 1e7


def test_rotated_grid_projection_is_rotated_pole() -> None:
    crs = CRS.from_cf(
        {
            "grid_mapping_name": "rotated_latitude_longitude",
            "grid_north_pole_latitude": 37.5,
            "grid_north_pole_longitude": 177.5,
        }
    )
    rotated = projection(Grid(np.linspace(-5, 5, 20), np.linspace(-5, 5, 30), crs))
    assert isinstance(rotated, ccrs.RotatedPole)
    assert rotated.proj4_params["o_lat_p"] == 37.5


@pytest.mark.parametrize("crs", ["EPSG:4326", "OGC:CRS84", "+proj=longlat +R=6371229"])
def test_plain_latlon_projection_is_plate_carree(crs: str) -> None:
    assert isinstance(projection(Grid([0.0, 1.0], [0.0, 1.0], crs)), ccrs.PlateCarree)


def test_sphere_projection_keeps_radius() -> None:
    globe = projection(Grid([0.0, 1.0], [0.0, 1.0], "+proj=longlat +R=6371229")).globe
    assert globe is not None
    assert globe.semimajor_axis == globe.semiminor_axis == 6371229


@pytest.mark.parametrize(
    "crs",
    [
        "+proj=longlat +ellps=clrk80ign +pm=paris",
        "+proj=laea +lat_0=52 +lon_0=10 +x_0=4321000 +y_0=3210000 +pm=paris",
        CRS.from_cf(
            {
                "grid_mapping_name": "rotated_latitude_longitude",
                "grid_north_pole_latitude": 37.5,
                "grid_north_pole_longitude": 177.5,
                "longitude_of_prime_meridian": 2.337229167,
            }
        ),
    ],
)
def test_non_greenwich_prime_meridian_is_rejected(crs: str | CRS) -> None:
    with pytest.raises(ValueError, match="prime meridian"):
        projection(Grid([0.0, 1.0], [0.0, 1.0], crs))
