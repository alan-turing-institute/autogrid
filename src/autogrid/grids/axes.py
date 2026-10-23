import numpy as np
from numpy.typing import ArrayLike, NDArray
from pyproj import CRS

from autogrid.crs import FULL_TURN


def check_axis_points(points: ArrayLike, axis: str) -> NDArray[np.float64]:
    """Return the points as a new read-only float64 array, checking they define an axis.

    The array is read-only so that bounds cached from it cannot go stale.

    Args:
        points: Coordinates of the points along the axis.
        axis: Name of the axis for error messages, e.g. "x".

    Returns:
        The points as a read-only 1D float64 array.

    Raises:
        ValueError: If the points are not 1D, have fewer than two points, contain
            non-finite values or are not strictly monotonic.
    """
    points = np.array(points, dtype=np.float64)
    if points.ndim != 1 or len(points) < 2:
        msg = (
            f"{axis} points must be a 1D array of at least two points, but got "
            f"shape {points.shape}."
        )
        raise ValueError(msg)
    if not np.isfinite(points).all():
        msg = f"{axis} points must all be finite."
        raise ValueError(msg)
    steps = np.diff(points)
    if not ((steps > 0).all() or (steps < 0).all()):
        msg = f"{axis} points must be strictly increasing or strictly decreasing."
        raise ValueError(msg)
    points.flags.writeable = False
    return points


def guess_bounds(points: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return (n, 2) cell bounds halfway between neighbouring points.

    The outer bounds are half a spacing beyond the first and last points.

    Args:
        points: 1D array of the points along an axis.

    Returns:
        2D array of the bounds of each cell along the axis, with shape (n, 2).
    """
    first, last = 2 * points[0] - points[1], 2 * points[-1] - points[-2]
    padded = np.concatenate([[first], points, [last]])
    edges = (padded[:-1] + padded[1:]) / 2
    return np.column_stack([edges[:-1], edges[1:]])


def normalise_longitude(
    x: ArrayLike, x_bounds: NDArray[np.float64], crs: CRS
) -> NDArray[np.float64]:
    """Shift x by whole turns of longitude to match cells with the given bounds.

    Args:
        x: x coordinates in the CRS.
        x_bounds: Bounds of the cells to match, in the CRS.
        crs: The CRS of x and x_bounds.

    Returns:
        For a geographic CRS, x shifted by whole turns to within half a turn of
        the middle of the cells; otherwise x unchanged.
    """
    x = np.asarray(x, dtype=np.float64)
    if not crs.is_geographic:
        return x
    middle = (np.min(x_bounds) + np.max(x_bounds)) / 2
    return middle + (x - middle + FULL_TURN / 2) % FULL_TURN - FULL_TURN / 2
