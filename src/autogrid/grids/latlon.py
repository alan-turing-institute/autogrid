import numpy as np

from autogrid.grid import Grid


def build_global_latlon(resolution: float) -> Grid:
    """Build a global regular latitude-longitude grid on WGS84 (EPSG:4326).

    Longitudes run from -180 to 180 and latitudes from -90 to 90, with grid
    points at cell centres.

    Args:
        resolution: Cell size in degrees. Must divide 180 exactly.

    Raises:
        ValueError: If the resolution is not in (0, 180] degrees, or does
            not divide 180 degrees exactly.
    """
    if not 0 < resolution <= 180:
        msg = f"Resolution {resolution} must be in (0, 180] degrees."
        raise ValueError(msg)
    if not np.isclose(180 / resolution, round(180 / resolution)):
        msg = f"Resolution {resolution} does not divide 180 degrees exactly."
        raise ValueError(msg)
    n_lon = round(360 / resolution)
    n_lat = round(180 / resolution)
    lon_points = -180 + (np.arange(n_lon) + 0.5) * resolution
    lat_points = -90 + (np.arange(n_lat) + 0.5) * resolution
    name = f"Global lat-lon {resolution:g} degree"
    return Grid(lon_points, lat_points, "EPSG:4326", name=name)
