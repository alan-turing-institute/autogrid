"""Regridding methods, each building the weights from one set of Locations to another.

Sources and targets are Locations: Grids, or Points of any shape. Methods that rely on
the rectilinear structure of a Grid raise a TypeError for other Locations. A method
returns a sparse matrix with shape (target size, source size), whose rows give
the weight of each source point in each target point, both flattened. Target points
outside the source cells of a Grid source have empty rows. The Regridder drops explicitly zero weights, so
that only the source points that contribute to a target point can make it NaN, and a
method need not drop them itself.

A Regridder takes either the name of one of these methods or any function like them. To
add a named method, write a function like these and add it to Method and METHODS.
"""

from collections.abc import Callable
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from autogrid.crs import FULL_TURN, MAX_LATITUDE
from autogrid.grids import Grid
from autogrid.locations import Locations
from autogrid.points import nearest_indices

Method = Literal["linear", "nearest", "nearest_great_circle", "area_weighted"]

# Builds the regridding weights from source Locations to target Locations
WeightsFunction = Callable[[Locations, Locations], sparse.csr_array]

# Fraction of a target cell below which area weighting ignores an overlap
MIN_OVERLAP_FRACTION = 1e-8


def area_weighted(source: Locations, target: Locations) -> sparse.csr_array:
    """Mean of the source cells over each target cell, weighted by their overlap.

    Cell areas are measured in the shared CRS, on a sphere for a geographic CRS. Target
    cells that the source cells do not wholly cover are outside, and overlaps with less
    than MIN_OVERLAP_FRACTION of a target cell are ignored.

    Args:
        source: Grid the input arrays are on.
        target: Grid to regrid onto.

    Returns:
        A sparse matrix with shape (target.size, source.size) giving the weight of each
        source point in each target point.

    Raises:
        TypeError: If the source or target is not a Grid.
        ValueError: If the grids do not share a CRS.
    """
    source = _require_grid(source, "area_weighted", "source")
    target = _require_grid(target, "area_weighted", "target")
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
            np.sin(np.deg2rad(np.clip(source.y_bounds, -MAX_LATITUDE, MAX_LATITUDE))),
            np.sin(np.deg2rad(np.clip(target.y_bounds, -MAX_LATITUDE, MAX_LATITUDE))),
        )
    else:
        x = _overlaps(source.x_bounds, target.x_bounds)
        y = _overlaps(source.y_bounds, target.y_bounds)
    weights = cast(sparse.csr_array, sparse.kron(y, x, format="csr"))
    uncovered = weights.sum(axis=1) < 1 - MIN_OVERLAP_FRACTION
    weights.data[np.repeat(uncovered, np.diff(weights.indptr))] = 0
    weights.data[weights.data < MIN_OVERLAP_FRACTION] = 0
    return weights


def linear(source: Locations, target: Locations) -> sparse.csr_array:
    """Bilinear interpolation in the source grid's own coordinates.

    Target points between the outermost source points and the edge of the source cells
    take the value of the edge cell.

    Args:
        source: Grid the input arrays are on.
        target: Locations to regrid onto.

    Returns:
        A sparse matrix with shape (target.size, source.size) giving the weight of each
        source point in each target point.

    Raises:
        TypeError: If the source is not a Grid.
    """
    source = _require_grid(source, "linear", "source")
    rows, y, x, _, _ = _find_target_points(source, target)
    x0, x1, fx = _bracket(source.x_points, x, circular=source.is_circular)
    y0, y1, fy = _bracket(source.y_points, y)
    nx = source.shape[1]
    cols = [y0 * nx + x0, y0 * nx + x1, y1 * nx + x0, y1 * nx + x1]
    weights = [(1 - fy) * (1 - fx), (1 - fy) * fx, fy * (1 - fx), fy * fx]
    # Each target point gets an entry for the four source points around it
    return sparse.csr_array(
        (np.concatenate(weights), (np.tile(rows, 4), np.concatenate(cols))),
        shape=(target.size, source.size),
    )


def nearest(source: Locations, target: Locations) -> sparse.csr_array:
    """Nearest source point in the source grid's own coordinates.

    Ties go to the source point with the lower coordinate, as in iris.

    Args:
        source: Grid the input arrays are on.
        target: Locations to regrid onto.

    Returns:
        A sparse matrix with shape (target.size, source.size) giving the weight of each
        source point in each target point.

    Raises:
        TypeError: If the source is not a Grid.
    """
    source = _require_grid(source, "nearest", "source")
    rows, y, x, _, _ = _find_target_points(source, target)
    x0, x1, fx = _bracket(source.x_points, x, circular=source.is_circular)
    y0, y1, fy = _bracket(source.y_points, y)
    # Each target point gets an entry for the nearest source point
    cols = np.where(fy > 0.5, y1, y0) * source.shape[1] + np.where(fx > 0.5, x1, x0)
    return sparse.csr_array(
        (np.ones(len(rows)), (rows, cols)), shape=(target.size, source.size)
    )


def nearest_great_circle(source: Locations, target: Locations) -> sparse.csr_array:
    """Nearest source point along the Earth's surface, from points.nearest_indices.

    For a Grid source, as for linear, target points outside the source cells in the
    source CRS get no weights. Other sources have no cells, so every target point takes
    its nearest source point, however far away. Either way, target points without a
    latitude or longitude get no weights.

    Args:
        source: Locations the input arrays are on.
        target: Locations to regrid onto.

    Returns:
        A sparse matrix with shape (target.size, source.size) giving the weight of each
        source point in each target point.
    """
    if isinstance(source, Grid):
        rows, _, _, flat_target_lats, flat_target_lons = _find_target_points(
            source, target
        )
    else:
        flat_lats, flat_lons = (coords.ravel() for coords in target.latlon())
        rows = np.flatnonzero(np.isfinite(flat_lats) & np.isfinite(flat_lons))
        flat_target_lats, flat_target_lons = flat_lats[rows], flat_lons[rows]
    if not len(rows):
        return sparse.csr_array((target.size, source.size))
    flat_source_lats, flat_source_lons = (coords.ravel() for coords in source.latlon())
    # Exclude grid points beyond the valid area of a projection
    flat_source_indices = np.flatnonzero(
        np.isfinite(flat_source_lats) & np.isfinite(flat_source_lons)
    )
    (nearest,) = nearest_indices(
        source_lats=flat_source_lats[flat_source_indices],
        source_lons=flat_source_lons[flat_source_indices],
        target_lats=flat_target_lats,
        target_lons=flat_target_lons,
    )
    return sparse.csr_array(
        (np.ones(len(rows)), (rows, flat_source_indices[nearest])),
        shape=(target.size, source.size),
    )


def _require_grid(locations: Locations, method: str, role: str) -> Grid:
    """Return the locations as a Grid, for methods that need one.

    Args:
        locations: Source or target of a regridding method.
        method: Name of the method, for the error message.
        role: "source" or "target", for the error message.

    Returns:
        The locations, as a Grid.

    Raises:
        TypeError: If the locations are not a Grid.
    """
    if not isinstance(locations, Grid):
        msg = (
            f"The {method!r} method needs a Grid {role}, but got {locations!r}. Use "
            '"nearest_great_circle" for other Locations.'
        )
        raise TypeError(msg)
    return locations


def _bracket(
    points: NDArray[np.float64], values: NDArray[np.float64], *, circular: bool = False
) -> tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.float64]]:
    """Find the neighbouring points either side of each value along one axis.

    Values beyond the outermost points are clamped to them. A circular axis wraps from
    its last point to its first, so values must already be shifted to within a turn of
    the points.

    Args:
        points: 1D array of the points along the axis.
        values: Coordinates along the axis to bracket, of any shape.
        circular: Whether the axis is circular, e.g. longitude.

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
    source: Grid, target: Locations
) -> tuple[
    NDArray[np.intp],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Find the target points inside the source cells, in the source CRS.

    Args:
        source: Grid the input arrays are on.
        target: Locations to regrid onto.

    Returns:
        Tuple of the flat target index of each point inside, its y and x in the source
        CRS, with x shifted by whole turns to match the source cells, and its latitude
        and longitude.
    """
    flat_lats, flat_lons = (coords.ravel() for coords in target.latlon())
    y, x = source.to_yx(flat_lats, flat_lons)
    inside = np.flatnonzero(source.contains(y=y, x=x))
    return inside, y[inside], x[inside], flat_lats[inside], flat_lons[inside]


def _overlaps(
    source_bounds: NDArray[np.float64],
    target_bounds: NDArray[np.float64],
    period: float | None = None,
) -> sparse.csr_array:
    """Fraction of each target cell covered by each source cell along one axis.

    Args:
        source_bounds: (n, 2) bounds of the source cells.
        target_bounds: (m, 2) bounds of the target cells.
        period: Period of the axis, e.g. a full turn of longitude, if cells overlap
            modulo the period.

    Returns:
        A sparse matrix with shape (m, n) giving the fraction of each target cell
        covered by each source cell.
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
    # Source cells are sorted so those overlapping a given target cell are contiguous
    # For each target cell, start[idx_target] is the index of the first overlapping cell
    # in this sorted array and counts[idx_target] is the number of overlapping cells.
    start = np.searchsorted(s_hi, t_lo, side="right")
    counts = np.maximum(np.searchsorted(s_lo, t_hi, side="left") - start, 0)
    # The target cell of each overlap, e.g. counts [2, 0, 3] give [0, 0, 2, 2, 2]
    target_overlap_indices = np.repeat(np.arange(len(t_lo)), counts)
    # The sorted source cell of each overlap, e.g. start [1, 4, 0] and counts [2, 0, 3]
    # give [1, 2, 0, 1, 2]
    source_overlap_indices = np.repeat(
        start - np.cumsum(counts) + counts, counts
    ) + np.arange(counts.sum())
    # Now calculate the overlap of each source cell with its target cell
    overlap = np.minimum(
        t_hi[target_overlap_indices], s_hi[source_overlap_indices]
    ) - np.maximum(t_lo[target_overlap_indices], s_lo[source_overlap_indices])
    return sparse.csr_array(
        (
            overlap / (t_hi - t_lo)[target_overlap_indices],
            (target_overlap_indices, cols[source_overlap_indices]),
        ),
        shape=(len(t_lo), len(source_bounds)),
    )


# Builds the weights of each regridding method for a (source, target) pair
METHODS: dict[Method, WeightsFunction] = {
    "linear": linear,
    "nearest": nearest,
    "nearest_great_circle": nearest_great_circle,
    "area_weighted": area_weighted,
}
