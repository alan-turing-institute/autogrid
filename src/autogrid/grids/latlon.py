import numpy as np

from autogrid.grid import Grid


class LatLonGrid(Grid):
    """A global regular latitude-longitude grid on WGS84 (EPSG:4326).

    Longitudes run from -180 to 180 and latitudes from -90 to 90, with grid
    points at cell centres.
    """

    def __init__(self, resolution: float) -> None:
        """Initialise a LatLonGrid.

        Args:
            resolution: Cell size in degrees. Must divide 180 exactly.
        """
        if not np.isclose(180 / resolution, round(180 / resolution)):
            msg = f"Resolution {resolution} does not divide 180 degrees exactly."
            raise ValueError(msg)
        self.resolution = resolution
        n_lon = round(360 / resolution)
        n_lat = round(180 / resolution)
        lon_points = -180 + (np.arange(n_lon) + 0.5) * resolution
        lat_points = -90 + (np.arange(n_lat) + 0.5) * resolution
        super().__init__(lon_points, lat_points, "EPSG:4326")

    def __repr__(self) -> str:
        return f"{type(self).__name__}(resolution={self.resolution})"
