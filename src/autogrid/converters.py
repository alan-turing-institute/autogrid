"""Convert between a Grid and unstructured points, by great circle distance.

These wrap the functions in autogrid.points, taking the grid's latitudes and
longitudes from a Grid.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from autogrid.grids import Grid
from autogrid.points import Points, gaussian_weighted_mean, nearest_indices


def grid_to_points(
    data: ArrayLike,
    *,
    grid: Grid,
    points: Points,
) -> NDArray[np.generic]:
    """Take the value of the nearest grid point to each point.

    This wraps autogrid.points.nearest_indices. Each point takes its nearest grid
    point, however far away, so nothing is masked outside the grid.

    Args:
        data: Array whose last two dimensions match the grid shape.
        grid: Grid the data are on.
        points: Points to sample the grid at.

    Returns:
        An array with the leading dimensions of data followed by the shape of points,
        and the dtype of data.

    Raises:
        ValueError: If the data shape does not end with the grid shape.
    """
    data = np.asarray(data)
    if data.shape[-2:] != grid.shape:
        msg = f"Data shape {data.shape} does not end with the grid shape {grid.shape}."
        raise ValueError(msg)
    latitudes, longitudes = grid.latlon()
    # Exclude grid points beyond the valid area of a projection
    has_latlon = np.flatnonzero(np.isfinite(latitudes) & np.isfinite(longitudes))
    (nearest,) = nearest_indices(
        source_lats=latitudes.ravel()[has_latlon],
        source_lons=longitudes.ravel()[has_latlon],
        target_lats=points.latitudes,
        target_lons=points.longitudes,
    )
    return data.reshape(*data.shape[:-2], grid.size)[..., has_latlon[nearest]]


def points_to_grid(
    data: ArrayLike,
    *,
    distance_scale_km: float,
    grid: Grid,
    points: Points,
) -> NDArray[np.floating]:
    """Interpolate scattered observations onto a grid with Gaussian weights.

    This wraps autogrid.points.gaussian_weighted_mean, which describes the
    weights. Every observation contributes to every grid point, so grid points
    far from all observations take the values of the nearest ones.

    Args:
        data: Observed values, whose trailing dimensions match the shape of points,
            e.g. (n_variables, n_obs). NaN or infinite observations are ignored.
            Only NaN marks missing values, so fill a masked array with
            data.filled(np.nan) first.
        distance_scale_km: Standard deviation of the Gaussian weights in km.
        grid: Grid to interpolate onto.
        points: Points where the observations were made.

    Returns:
        An array with the leading dimensions of data followed by the grid shape.
        Grid points with no valid observations, or with no latitude or longitude,
        are NaN.

    Raises:
        ValueError: As for gaussian_weighted_mean.
    """
    flat_latitudes, flat_longitudes = (coords.ravel() for coords in grid.latlon())
    # Exclude grid points beyond the valid area of a projection
    has_latlon = np.isfinite(flat_latitudes) & np.isfinite(flat_longitudes)
    gridded = gaussian_weighted_mean(
        source_lats=points.latitudes,
        source_lons=points.longitudes,
        data=data,
        target_lats=flat_latitudes[has_latlon],
        target_lons=flat_longitudes[has_latlon],
        distance_scale_km=distance_scale_km,
    )
    result = np.full((*gridded.shape[:-1], grid.size), np.nan, dtype=gridded.dtype)
    result[..., has_latlon] = gridded
    return result.reshape(*gridded.shape[:-1], *grid.shape)
