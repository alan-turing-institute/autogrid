import cartopy.crs as ccrs
import numpy as np
import pytest
from pyproj import CRS

from autogrid import Grid
from autogrid.grids import build_ease2
from autogrid.grids.ease2 import Region

PARIS = "+proj=longlat +ellps=clrk80ign +pm=paris +no_defs"
PARIS_DEGREES = 2.5969213 * 0.9


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


@pytest.mark.parametrize("crs", [PARIS, "EPSG:4807"])
def test_projection_respects_prime_meridian(crs: str) -> None:
    projection = Grid([0.0, 1.0], [45.0, 46.0], crs).projection
    lon, _ = ccrs.PlateCarree().transform_point(0.0, 45.0, projection)
    np.testing.assert_allclose(lon, PARIS_DEGREES, atol=1e-2)


@pytest.mark.parametrize("crs", ["EPSG:4326", "OGC:CRS84", "+proj=longlat +R=6371229"])
def test_plain_latlon_projection_is_plate_carree(crs: str) -> None:
    assert isinstance(Grid([0.0, 1.0], [0.0, 1.0], crs).projection, ccrs.PlateCarree)


def test_projection_keeps_datum() -> None:
    # OSGB 1936 is offset from WGS 84 by about 100 m
    projection = Grid([0.0, 1.0], [52.0, 53.0], "EPSG:4277").projection
    assert not isinstance(projection, ccrs.PlateCarree)
    lon, _ = ccrs.CRS("EPSG:4326").transform_point(-2.0, 52.0, projection)
    assert abs(lon + 2.0) > 1e-3


@pytest.mark.parametrize(
    ("units", "metres"), [("m", 1.0), ("km", 1000.0), ("us-ft", 1200 / 3937)]
)
def test_laea_projection_keeps_linear_units(units: str, metres: float) -> None:
    crs = f"+proj=laea +lat_0=90 +lon_0=0 +x_0=1000 +datum=WGS84 +units={units}"
    projection = Grid([0.0, 1.0], [0.0, 1.0], crs).projection
    x, y = projection.transform_point(0.0, 80.0, ccrs.PlateCarree())
    expected = CRS(crs).to_cf()["false_easting"]
    np.testing.assert_allclose(x, expected, rtol=1e-9)
    np.testing.assert_allclose(y * metres, -1111950, rtol=1e-2)
    x_min, x_max = projection.x_limits
    assert (x_max - x_min) * metres > 1e7
