import numpy as np

from autogrid.crs import MAX_LATITUDE, MAX_LONGITUDE
from autogrid.grids.grid import Grid


def build_global_latlon(resolution: float) -> Grid:
    """Build a global regular latitude-longitude grid on WGS84 (EPSG:4326).

    Longitudes run from -180 to 180 and latitudes from -90 to 90, with grid
    points at cell centres.

    Args:
        resolution: Cell size in degrees. Must divide 180 exactly.

    Raises:
        ValueError: If the resolution is not in (0, 180) degrees, or does
            not divide 180 degrees exactly.
    """
    if not 0 < resolution < 2 * MAX_LATITUDE:
        msg = f"Resolution {resolution} must be in (0, 180) degrees."
        raise ValueError(msg)
    lat_cells = 2 * MAX_LATITUDE / resolution
    if not np.isclose(lat_cells, round(lat_cells)):
        msg = f"Resolution {resolution} does not divide 180 degrees exactly."
        raise ValueError(msg)
    n_lon = round(2 * MAX_LONGITUDE / resolution)
    n_lat = round(lat_cells)
    lon_points = -MAX_LONGITUDE + (np.arange(n_lon) + 0.5) * resolution
    lat_points = -MAX_LATITUDE + (np.arange(n_lat) + 0.5) * resolution
    name = f"Global lat-lon {resolution:g} degree"
    return Grid(lon_points, lat_points, "EPSG:4326", name=name)
