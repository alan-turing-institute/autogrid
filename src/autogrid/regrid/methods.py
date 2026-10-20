"""Regridding methods, each building the weights from one Grid to another.

A method returns a sparse matrix with shape (target size, source size), whose rows give
the weight of each source point in each target point, both flattened. Target points
outside the source grid have empty rows, and weights are never explicitly zero, so that
only the source points that contribute to a target point can make it NaN.

A Regridder takes either the name of one of these methods or any function like them. To
add a named method, write a function like these and add it to Method and METHODS.
"""

from collections.abc import Callable
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from autogrid.crs import trf_from_lonlat
from autogrid.grids import Grid
from autogrid.grids.axes import FULL_TURN, normalise_longitude
from autogrid.points import nearest_indices

Method = Literal["linear", "nearest", "nearest_great_circle", "area_weighted"]

# Builds the regridding weights from a source Grid to a target Grid
WeightsFunction = Callable[[Grid, Grid], sparse.csr_array]

# Fraction of a target cell below which area weighting ignores an overlap
MIN_OVERLAP_FRACTION = 1e-8


def area_weighted(source: Grid, target: Grid) -> sparse.csr_array:
    """Mean of the source cells over each target cell, weighted by their overlap.

    Cell areas are measured in the shared CRS, on a sphere for a geographic CRS. Target
    cells that the source cells do not wholly cover are outside, and overlaps with less
    than MIN_OVERLAP_FRACTION of a target cell are ignored.

    Raises:
        ValueError: If the grids do not share a CRS.
    """
    if not source.crs.equals(target.crs, ignore_axis_order=True):
        msg = (
            "Area weighted regridding requires both grids to share a CRS, but got "
            f"{source.crs.to_string()!r} and {target.crs.to_string()!r}."
        )
        raise ValueError(msg)
    if source.is_geographic:
        x = _overlaps(source.x_bounds, target.x_bounds, period=FULL_TURN)
        # Cell areas on a sphere are proportional to the change in sin(latitude)
        y = _overlaps(
            np.sin(np.deg2rad(np.clip(source.y_bounds, -90.0, 90.0))),
            np.sin(np.deg2rad(np.clip(target.y_bounds, -90.0, 90.0))),
        )
    else:
        x = _overlaps(source.x_bounds, target.x_bounds)
        y = _overlaps(source.y_bounds, target.y_bounds)
    weights = cast(sparse.csr_array, sparse.kron(y, x, format="csr"))
    uncovered = weights.sum(axis=1) < 1 - MIN_OVERLAP_FRACTION
    weights.data[np.repeat(uncovered, np.diff(weights.indptr))] = 0
    weights.data[weights.data < MIN_OVERLAP_FRACTION] = 0
    weights.eliminate_zeros()
    return weights


def linear(source: Grid, target: Grid) -> sparse.csr_array:
    """Bilinear interpolation in the source grid's own coordinates.

    Target points between the outermost source points and the edge of the source cells
    take the value of the edge cell.
    """
    rows, x, y, _, _ = _find_target_points(source, target)
    x0, x1, fx = _bracket(source.x_points, x, circular=source.is_circular)
    y0, y1, fy = _bracket(source.y_points, y)
    nx = source.shape[1]
    cols = [y0 * nx + x0, y0 * nx + x1, y1 * nx + x0, y1 * nx + x1]
    weights = [(1 - fy) * (1 - fx), (1 - fy) * fx, fy * (1 - fx), fy * fx]
    # Each target point gets an entry for the four source points around it
    matrix = sparse.csr_array(
        (np.concatenate(weights), (np.tile(rows, 4), np.concatenate(cols))),
        shape=(target.size, source.size),
    )
    # Drop zero weights if there are any
    matrix.eliminate_zeros()
    return matrix


def nearest(source: Grid, target: Grid) -> sparse.csr_array:
    """Nearest source point in the source grid's own coordinates.

    Ties go to the source point with the lower coordinate, as in iris.
    """
    rows, x, y, _, _ = _find_target_points(source, target)
    x0, x1, fx = _bracket(source.x_points, x, circular=source.is_circular)
    y0, y1, fy = _bracket(source.y_points, y)
    # Each target point gets an entry for the nearest source point
    cols = np.where(fy > 0.5, y1, y0) * source.shape[1] + np.where(fx > 0.5, x1, x0)
    return sparse.csr_array(
        (np.ones(len(rows)), (rows, cols)), shape=(target.size, source.size)
    )


def nearest_great_circle(source: Grid, target: Grid) -> sparse.csr_array:
    """Nearest source point along the Earth's surface, from points.nearest_indices.

    Target points are outside as for linear, in the source CRS.
    """
    rows, _, _, target_lons, target_lats = _find_target_points(source, target)
    if not len(rows):
        return sparse.csr_array((target.size, source.size))
    source_lons, source_lats = source.lonlat_slice(slice(None))
    (cols,) = nearest_indices(
        source_lats.ravel(), source_lons.ravel(), target_lats, target_lons
    )
    return sparse.csr_array(
        (np.ones(len(rows)), (rows, cols)), shape=(target.size, source.size)
    )


def _bracket(
    points: NDArray[np.float64], values: NDArray[np.float64], *, circular: bool = False
) -> tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.float64]]:
    """Find the neighbouring points either side of each value along one axis.

    Values beyond the outermost points are clamped to them. A circular axis wraps from
    its last point to its first, so values must already be shifted to within a turn of
    the points.

    Returns:
        Tuple of (lower, upper, fraction), giving the indices of the points below and
        above each value, and the fraction of the way from the lower to the upper point.
    """
    index = np.arange(len(points))
    if points[0] > points[-1]:
        points, index = points[::-1], index[::-1]
    if circular:
        points = np.concatenate(
            [points[-1:] - FULL_TURN, points, points[:1] + FULL_TURN]
        )
        index = np.concatenate([index[-1:], index, index[:1]])
    i = np.clip(np.searchsorted(points, values) - 1, 0, len(points) - 2)
    fraction = np.clip((values - points[i]) / (points[i + 1] - points[i]), 0, 1)
    return index[i], index[i + 1], fraction


def _find_target_points(
    source: Grid, target: Grid
) -> tuple[
    NDArray[np.intp],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Find the target points inside the source cells, in the source CRS.

    Returns:
        Tuple of the flat target index of each point inside, its x and y in the source
        CRS, with x shifted by whole turns to match the source cells, and its longitude
        and latitude.
    """
    lon, lat = target.lonlat_slice(slice(None))
    lon, lat = lon.ravel(), lat.ravel()
    x, y = trf_from_lonlat(source.crs).transform(lon, lat)
    x = normalise_longitude(x, source.x_bounds, source.crs)
    rows = np.flatnonzero(source.contains(x, y))
    return rows, x[rows], y[rows], lon[rows], lat[rows]


def _overlaps(
    source_bounds: NDArray[np.float64],
    target_bounds: NDArray[np.float64],
    period: float | None = None,
) -> sparse.csr_array:
    """Fraction of each target cell covered by each source cell along one axis.

    Args:
        source_bounds: (n, 2) bounds of the source cells, which must be monotonic.
        target_bounds: (m, 2) bounds of the target cells, which must be monotonic.
        period: Period of the axis, e.g. a full turn of longitude, if cells overlap
            modulo the period.

    Returns:
        A sparse matrix with shape (m, n).
    """
    s_lo, s_hi = np.sort(source_bounds, axis=1).T
    t_lo, t_hi = np.sort(target_bounds, axis=1).T
    cols = np.arange(len(s_lo))
    if period is not None:
        # Start each target cell within one period above the source cells, and
        # repeat the source cells one period on to catch overlaps across the wrap
        shift = period * np.floor((t_lo - s_lo.min()) / period)
        t_lo, t_hi = t_lo - shift, t_hi - shift
        s_lo = np.concatenate([s_lo, s_lo + period])
        s_hi = np.concatenate([s_hi, s_hi + period])
        cols = np.tile(cols, 2)
    order = np.argsort(s_lo)
    s_lo, s_hi, cols = s_lo[order], s_hi[order], cols[order]
    # The source cells overlapping a target cell run from the first that ends
    # after it starts to the last that starts before it ends
    start = np.searchsorted(s_hi, t_lo, side="right")
    counts = np.maximum(np.searchsorted(s_lo, t_hi, side="left") - start, 0)
    rows = np.repeat(np.arange(len(t_lo)), counts)
    k = np.repeat(start - np.cumsum(counts) + counts, counts) + np.arange(counts.sum())
    overlap = np.minimum(t_hi[rows], s_hi[k]) - np.maximum(t_lo[rows], s_lo[k])
    return sparse.csr_array(
        (overlap / (t_hi - t_lo)[rows], (rows, cols[k])),
        shape=(len(t_lo), len(source_bounds)),
    )


# Builds the weights of each regridding method for a (source, target) pair
METHODS: dict[Method, WeightsFunction] = {
    "linear": linear,
    "nearest": nearest,
    "nearest_great_circle": nearest_great_circle,
    "area_weighted": area_weighted,
}
