from typing import Literal

import numpy as np
from iris.coord_systems import CoordSystem, LambertAzimuthalEqualArea

from autogrid.coord_systems import WGS84, EASE2GlobalCS
from autogrid.grid import Grid

Region = Literal["north", "south", "global"]

# Cell size (m) and (ny, nx) of each full EASE-Grid 2.0 grid, from the NSIDC spec
# https://nsidc.org/data/user-resources/help-center/guide-ease-grids
_POLAR = {
    "36km": (36000.0, (500, 500)),
    "25km": (25000.0, (720, 720)),
    "12p5km": (12500.0, (1440, 1440)),
    "10km": (9000.0, (1800, 1800)),
    "9km": (9000.0, (2000, 2000)),
    "6p25km": (6250.0, (2880, 2880)),
    "3p125km": (3125.0, (5760, 5760)),
    "3km": (3000.0, (6000, 6000)),
    "1km": (1000.0, (18000, 18000)),
}
_GLOBAL = {
    "36km": (36032.220840584, (406, 964)),
    "25km": (25025.26000, (584, 1388)),
    "12p5km": (12512.63000, (1168, 2776)),
    "9km": (9008.055210146, (1624, 3856)),
    "3km": (3002.690070049, (4872, 11568)),
    "1km": (1000.895023350, (14616, 34704)),
}
SPEC: dict[Region, dict[str, tuple[float, tuple[int, int]]]] = {
    "north": _POLAR,
    "south": _POLAR,
    "global": _GLOBAL,
}


def _coord_system(region: Region) -> CoordSystem:
    if region == "global":
        return EASE2GlobalCS()
    return LambertAzimuthalEqualArea(
        latitude_of_projection_origin=90.0 if region == "north" else -90.0,
        longitude_of_projection_origin=0.0,
        false_easting=0.0,
        false_northing=0.0,
        ellipsoid=WGS84,
    )


class EASE2Grid(Grid):
    """An EASE-Grid 2.0 grid (EPSG:6931 north, 6932 south, 6933 global).

    Rows run from top (+y) to bottom (-y), matching the NSIDC data files.
    """

    def __init__(
        self,
        region: Region,
        resolution: str,
        shape: tuple[int, int] | None = None,
    ) -> None:
        """Initialise an EASE2Grid.

        Args:
            region: "north", "south" or "global".
            resolution: Nominal resolution, e.g. "25km".
            shape: Optional (ny, nx) of a subset centred on the projection
                origin. Each dimension must have the same parity as the full
                grid so that cells align with the official grid. Defaults to
                the full grid.
        """
        cell, full_shape = SPEC[region][resolution]
        shape = shape or full_shape
        for n, n_full in zip(shape, full_shape, strict=True):
            if n > n_full or (n_full - n) % 2:
                msg = f"Shape {shape} is not a centred subset of {full_shape}."
                raise ValueError(msg)
        self.region = region
        self.resolution = resolution
        ny, nx = shape
        x_points = (np.arange(nx) - (nx - 1) / 2) * cell
        y_points = ((ny - 1) / 2 - np.arange(ny)) * cell
        super().__init__(x_points, y_points, _coord_system(region))

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(region={self.region!r}, "
            f"resolution={self.resolution!r}, shape={self.shape})"
        )
