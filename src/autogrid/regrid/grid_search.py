"""Nearest neighbour search by great circle distance, on the points of a Grid.

This walks the source grid from a first guess in its own CRS, rather than
building a KD-tree over all of its points as points.nearest_indices does.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from autogrid.grids import Grid
from autogrid.grids.axes import normalise_longitude
from autogrid.regrid.sphere import check_points, unit_vectors

# Largest number of target points whose latitudes and longitudes are held at
# once by grid_to_grid
_BLOCK_POINTS = 2**22

# Offsets of the eight neighbours of a grid cell, as (row, column)
_NEIGHBOURS = np.array(
    [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]
)


def grid_to_points(
    source: Grid, target_lats: ArrayLike, target_lons: ArrayLike
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Source grid indices of the nearest source point to each target point.

    This finds the same nearest points as points.nearest_indices(
    source.latitudes, source.longitudes, ...), but uses the structure of the
    source grid rather than a KD-tree over all of its points. It needs memory
    in proportion to the number of target points only, so suits very large
    source grids, such as the 1 km EASE-Grid 2.0 grids.

    Each target point starts from the source point nearest to it in the source
    CRS, then moves to whichever of its eight neighbours is nearer by great
    circle distance, until none are. This finds the true nearest point wherever
    the projection is smooth on the scale of a few cells, which holds for
    target points within the source grid. Target points outside it, or that
    cannot be projected into the source CRS, first search with steps of many
    cells, halving them whenever no neighbour is nearer. The point found is
    then nearest along the edge of the grid. For target points more than 90
    degrees from the source grid, much of its edge can be at similar
    distances, and this may not be the nearest point overall.

    Args:
        source: Grid of the source points.
        target_lats: Latitude of each target point in degrees, of any shape.
        target_lons: Longitude of each target point in degrees, of the same shape.

    Returns:
        Tuple of (iy, ix) integer arrays with the target shape, giving the row
        and column of the nearest source point.

    Raises:
        ValueError: If the target latitudes and longitudes differ in shape, are
            empty, or are not all finite.
    """
    target_lats, target_lons = check_points("Target", target_lats, target_lons)
    lats, lons = target_lats.ravel(), target_lons.ravel()
    x, y = source.trf_from_latlon.transform(lons, lats)
    x = normalise_longitude(x, source.x_bounds, source.crs)
    iy = _nearest_index(source.y_points, y)
    ix = _nearest_index(source.x_points, x)
    ny, nx = source.shape
    # Search in steps of a quarter of the grid from points outside it, as they
    # can start far from their nearest point
    inside = source.contains(x, y)
    step = np.where(inside, 1, max(1, 1 << (max(ny, nx).bit_length() - 3)))

    targets = unit_vectors(lats, lons)

    def distance(
        targets: NDArray[np.float64], iy: NDArray[np.intp], ix: NDArray[np.intp]
    ) -> NDArray[np.float64]:
        """Squared straight-line distance between unit vectors, as in a KD-tree."""
        # Grid.latitudes and Grid.longitudes come from this transform, so
        # distances match those of the KD-tree search exactly
        lon, lat = source.trf_to_latlon.transform(
            source.x_points[ix], source.y_points[iy]
        )
        vectors = unit_vectors(lat, lon).reshape(*iy.shape, 3)
        return ((vectors - targets) ** 2).sum(axis=-1)

    active = np.arange(lats.size)
    best = distance(targets, iy, ix)
    while len(active):
        # Move each active point to its nearest neighbour, if that is nearer,
        # or else search nearer neighbours, until there are none to search
        offsets = _NEIGHBOURS * step[active, np.newaxis, np.newaxis]
        y = np.clip(iy[active, np.newaxis] + offsets[..., 0], 0, ny - 1)
        x = ix[active, np.newaxis] + offsets[..., 1]
        x = x % nx if source.is_circular else np.clip(x, 0, nx - 1)
        d = distance(targets[active, np.newaxis], y, x)
        k = np.argmin(d, axis=1)
        rows = np.arange(len(active))
        moved = d[rows, k] < best[active]
        step[active[~moved]] //= 2
        a, k, rows = active[moved], k[moved], rows[moved]
        iy[a], ix[a], best[a] = y[rows, k], x[rows, k], d[rows, k]
        active = active[step[active] > 0]
    return iy.reshape(target_lats.shape), ix.reshape(target_lats.shape)


def grid_to_grid(
    source: Grid, target: Grid
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Source grid indices of the nearest source point to each target grid point.

    As grid_to_points, for the latitudes and longitudes of the target grid.
    These are found a block of rows at a time, so that neither grid holds the
    latitudes and longitudes of all of its points at once. Regridder uses this
    for its "nearest_great_circle" method.

    Args:
        source: Grid of the source points.
        target: Grid of the target points.

    Returns:
        Tuple of (iy, ix) integer arrays with the target grid shape (ny, nx),
        giving the row and column of the nearest source point.
    """
    iy = np.empty(target.shape, dtype=np.intp)
    ix = np.empty(target.shape, dtype=np.intp)
    rows = max(1, _BLOCK_POINTS // target.shape[1])
    for start in range(0, target.shape[0], rows):
        block = slice(start, start + rows)
        iy[block], ix[block] = grid_to_points(source, *target.latlon_slice(block))
    return iy, ix


def _nearest_index(
    points: NDArray[np.float64], values: NDArray[np.float64]
) -> NDArray[np.intp]:
    """Index of the nearest of monotonic points to each value, clamped to the ends."""
    if points[0] > points[-1]:
        return len(points) - 1 - _nearest_index(points[::-1], values)
    upper = np.clip(np.searchsorted(points, values), 1, len(points) - 1)
    nearer_lower = values - points[upper - 1] < points[upper] - values
    return np.where(nearer_lower, upper - 1, upper)
