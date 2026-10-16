"""Nearest neighbour search by great circle distance, on the points of a Grid.

This walks the source grid from a first guess in its own CRS, rather than
building a KD-tree over all of its points as points.nearest_indices does.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pyproj import Transformer
from scipy.spatial import KDTree

from autogrid.grids import Grid
from autogrid.grids.axes import normalise_longitude
from autogrid.regrid.sphere import check_points, unit_vectors

# Largest number of target points whose latitudes and longitudes are held at
# once by search_grid and outside_grid
_BLOCK_POINTS = 2**22

# Offsets of a grid cell and its eight neighbours, as (row, column), with the cell
# itself first so that it wins ties
_NEIGHBOURS = np.array(
    [(0, 0)]
    + [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]
)


def grid_to_points(
    source: Grid, target_lats: ArrayLike, target_lons: ArrayLike
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Source grid indices of the nearest source point to each target point.

    This finds the same nearest points as points.nearest_indices(source.latitudes,
    source.longitudes, ...), but uses the structure of the source grid rather than a
    KD-tree over all of its points. Memory usage scales with the number of target
    points and the length of the edge of the source grid only, so this is suitable for
    very large source grids, such as the 1 km EASE-Grid 2.0 grids.

    Each target point starts from the point nearest to it in the source grid. Each
    target point outside the grid area, or that cannot be projected into the source CRS,
    starts from the nearest point on the edge of the source grid, found with a KD-tree
    over the edge points only. Each then moves to whichever of its eight neighbours is
    nearer by great circle distance, until none are. This finds the true nearest point
    wherever the projection is smooth on the scale of a few cells.

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
    iy, ix, _ = _search(source, target_lats, target_lons)
    return iy, ix


def _search(
    source: Grid, target_lats: NDArray[np.float64], target_lons: NDArray[np.float64]
) -> tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.bool_]]:
    """As grid_to_points for checked targets, also flagging those outside the source.

    A target is outside when it starts its search outside the cells of the
    source grid.
    """
    lats, lons = target_lats.ravel(), target_lons.ravel()
    x, y = source.trf_from_latlon.transform(lons, lats)
    x = normalise_longitude(x, source.x_bounds, source.crs)
    iy = nearest_index(source.y_points, y)
    ix = nearest_index(source.x_points, x)
    ny, nx = source.shape
    targets = unit_vectors(lats, lons)
    # The nearest point to one outside the grid is usually on its edge, where a search
    # from the nearest point in the source CRS can stop at the wrong corner, so start
    # from the nearest edge point instead
    outside = ~source.contains(x, y, prenormalised=True)
    if outside.any():
        # Get the indices of the points on the edge of the grid
        rows, cols = np.arange(ny), np.arange(nx)
        edge_iy = np.concatenate(
            [np.zeros_like(cols), np.full_like(cols, ny - 1), rows, rows]
        )
        edge_ix = np.concatenate(
            [cols, cols, np.zeros_like(rows), np.full_like(rows, nx - 1)]
        )
        # Build a KD-tree over the edge points that have a latitude and longitude, and
        # find the nearest edge point to each
        vectors = point_vectors(source, edge_iy, edge_ix)
        valid = np.isfinite(vectors).all(-1)
        edge_iy, edge_ix = edge_iy[valid], edge_ix[valid]
        tree = KDTree(vectors[valid])
        _, k = tree.query(targets[outside], workers=-1)
        iy[outside], ix[outside] = edge_iy[k], edge_ix[k]

    active = np.arange(lats.size)
    while len(active):
        # Move each active point to whichever of its cell and eight neighbours is
        # nearest, as squared straight-line distance in a KD-tree, until that is its
        # own cell. Cells with no latitude and longitude, such as those beyond the
        # valid area of the source CRS, are never nearest
        y = np.clip(iy[active, np.newaxis] + _NEIGHBOURS[:, 0], 0, ny - 1)
        x = ix[active, np.newaxis] + _NEIGHBOURS[:, 1]
        x = x % nx if source.is_circular else np.clip(x, 0, nx - 1)
        d = ((point_vectors(source, y, x) - targets[active, np.newaxis]) ** 2).sum(-1)
        k = np.argmin(np.where(np.isnan(d), np.inf, d), axis=1)
        moved = k > 0
        active, rows, k = active[moved], np.flatnonzero(moved), k[moved]
        iy[active], ix[active] = y[rows, k], x[rows, k]
    shape = target_lats.shape
    return iy.reshape(shape), ix.reshape(shape), outside.reshape(shape)


def grid_to_grid(
    source: Grid, target: Grid
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Source grid indices of the nearest source point to each target grid point.

    As grid_to_points, for the latitudes and longitudes of the target grid. These are
    found a block of rows at a time, so that neither grid holds the latitudes and
    longitudes of all of its points at once. Regridder uses this for its
    "nearest_great_circle" method.

    Args:
        source: Grid of the source points.
        target: Grid of the target points.

    Returns:
        Tuple of (iy, ix) integer arrays with the target grid shape (ny, nx),
        giving the row and column of the nearest source point.
    """
    iy, ix, _ = search_grid(source, target)
    return iy, ix


def search_grid(
    source: Grid, target: Grid
) -> tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.bool_]]:
    """As grid_to_grid, also flagging target points outside the source grid.

    A target point is outside when its search starts outside the cells of the
    source grid, having reached the source CRS through its own latitude and
    longitude, with no datum shift between the grids.

    Returns:
        Tuple of (iy, ix, outside) arrays with the target grid shape (ny, nx).
    """
    iy = np.empty(target.shape, dtype=np.intp)
    ix = np.empty(target.shape, dtype=np.intp)
    outside = np.empty(target.shape, dtype=np.bool_)
    for block in row_blocks(target):
        # Grid latitudes and longitudes are always finite, so need no checks
        iy[block], ix[block], outside[block] = _search(
            source, *target.latlon_slice(block)
        )
    return iy, ix, outside


def nearest_index(
    points: NDArray[np.float64], values: NDArray[np.float64]
) -> NDArray[np.intp]:
    """Index of the nearest of monotonic points to each value, clamped to the ends."""
    if points[0] > points[-1]:
        return len(points) - 1 - nearest_index(points[::-1], values)
    upper = np.clip(np.searchsorted(points, values), 1, len(points) - 1)
    nearer_lower = values - points[upper - 1] < points[upper] - values
    return np.where(nearer_lower, upper - 1, upper)


def outside_grid(
    source: Grid, target: Grid, transformer: Transformer | None
) -> NDArray[np.bool_]:
    """Flag target grid points that lie outside the cells of the source grid.

    The target is processed a block of rows at a time.

    Args:
        source: Grid of the source points.
        target: Grid of the target points.
        transformer: Transformer from the target CRS to the source CRS, or None
            if the target points are already in the source CRS.

    Returns:
        A boolean array with the target grid shape (ny, nx).
    """
    outside = np.empty(target.shape, dtype=np.bool_)
    for block in row_blocks(target):
        x, y = target.x[block], target.y[block]
        if transformer is not None:
            x, y = transformer.transform(x, y)
        outside[block] = ~source.contains(x, y)
    return outside


def point_vectors(
    grid: Grid, iy: NDArray[np.intp], ix: NDArray[np.intp]
) -> NDArray[np.float64]:
    """Unit vectors of the grid points at rows iy and columns ix, of any shape.

    Grid.latitudes and Grid.longitudes come from the same transform, so distances
    match those of a KD-tree over them exactly.
    """
    lon, lat = grid.trf_to_latlon.transform(grid.x_points[ix], grid.y_points[iy])
    return unit_vectors(lat, lon).reshape(*iy.shape, 3)


def row_blocks(grid: Grid) -> list[slice]:
    """Blocks of rows of a grid, each with at most about _BLOCK_POINTS points."""
    rows = max(1, _BLOCK_POINTS // grid.shape[1])
    return [slice(start, start + rows) for start in range(0, grid.shape[0], rows)]
