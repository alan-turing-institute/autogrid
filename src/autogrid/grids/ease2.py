from typing import Literal

import numpy as np

from autogrid.grids.grid import Grid

Region = Literal["north", "south", "global"]

# Cell size (m) and (ny, nx) of each full EASE-Grid 2.0 grid, from the NSIDC spec
# https://nsidc.org/data/user-resources/help-center/guide-ease-grids
_POLAR = {
    "36p0km": (36000.0, (500, 500)),
    "25p0km": (25000.0, (720, 720)),
    "12p5km": (12500.0, (1440, 1440)),
    "10p0km": (10000.0, (1800, 1800)),
    "9p0km": (9000.0, (2000, 2000)),
    "6p25km": (6250.0, (2880, 2880)),
    "3p125km": (3125.0, (5760, 5760)),
    "3p0km": (3000.0, (6000, 6000)),
    "1p0km": (1000.0, (18000, 18000)),
}
_GLOBAL = {
    "36p0km": (36032.220840584, (406, 964)),
    "25p0km": (25025.26000, (584, 1388)),
    "12p5km": (12512.63000, (1168, 2776)),
    "9p0km": (9008.055210146, (1624, 3856)),
    "3p0km": (3002.690070049, (4872, 11568)),
    "1p0km": (1000.895023350, (14616, 34704)),
}
SPEC: dict[Region, dict[str, tuple[float, tuple[int, int]]]] = {
    "north": _POLAR,
    "south": _POLAR,
    "global": _GLOBAL,
}


# EPSG code of the EASE-Grid 2.0 CRS for each region
EPSG: dict[Region, int] = {"north": 6931, "south": 6932, "global": 6933}


def build_ease2(
    region: Region,
    resolution: str,
    shape: tuple[int, int] | None = None,
) -> Grid:
    """Build an EASE-Grid 2.0 grid (EPSG:6931 north, 6932 south, 6933 global).

    Rows run from top (+y) to bottom (-y), matching the NSIDC data files.

    Args:
        region: "north", "south" or "global".
        resolution: Nominal resolution, e.g. "25p0km".
        shape: Optional (ny, nx) of a subset centred on the projection
            origin. Each dimension must have the same parity as the full
            grid so that cells align with the official grid. Defaults to
            the full grid.

    Raises:
        ValueError: If the region or resolution is unknown, or the shape is
            not a non-empty centred subset of the full grid.
    """
    if region not in SPEC:
        msg = f"Unknown region {region!r}; expected one of {list(SPEC)}."
        raise ValueError(msg)
    if resolution not in SPEC[region]:
        msg = (
            f"Unknown resolution {resolution!r} for region {region!r}; "
            f"expected one of {list(SPEC[region])}."
        )
        raise ValueError(msg)
    cell, full_shape = SPEC[region][resolution]
    shape = shape or full_shape
    for n, n_full in zip(shape, full_shape, strict=True):
        if not 0 < n <= n_full or (n_full - n) % 2:
            msg = f"Shape {shape} is not a centred subset of {full_shape}."
            raise ValueError(msg)
    ny, nx = shape
    x_points = (np.arange(nx) - (nx - 1) / 2) * cell
    y_points = ((ny - 1) / 2 - np.arange(ny)) * cell
    return Grid(x_points, y_points, EPSG[region], name=f"EASE2 {region} {resolution}")
