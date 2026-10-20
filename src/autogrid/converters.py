"""Convert between a Grid and unstructured points, by great circle distance.

These wrap the functions in autogrid.points, taking the grid's latitudes and
longitudes from a Grid.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from autogrid.grids import Grid
from autogrid.points import gaussian_weighted_mean, nearest_indices


def grid_to_points(
    grid: Grid,
    data: ArrayLike,
    target_lats: ArrayLike,
    target_lons: ArrayLike,
) -> NDArray[np.generic]:
    """Take the value of the nearest grid point to each target point.

    This wraps autogrid.points.nearest_indices. Every target point takes its
    nearest grid point, however far away, so nothing is masked outside the grid.

    Args:
        grid: Grid the data are on.
        data: Array whose last two dimensions match the grid shape.
        target_lats: Latitude of each target point in degrees, of any shape.
        target_lons: Longitude of each target point in degrees, of the same shape.

    Returns:
        An array with the leading dimensions of data followed by the target shape,
        and the dtype of data.

    Raises:
        ValueError: If the data shape does not end with the grid shape, or if the
            target latitudes and longitudes differ in shape, are empty, or are not
            all finite.
    """
    data = np.asarray(data)
    if data.shape[-2:] != grid.shape:
        msg = f"Data shape {data.shape} does not end with the grid shape {grid.shape}."
        raise ValueError(msg)
    longitudes, latitudes = grid.lonlat()
    # Exclude grid points beyond the valid area of a projection
    has_latlon = np.flatnonzero(np.isfinite(latitudes) & np.isfinite(longitudes))
    (nearest,) = nearest_indices(
        latitudes.ravel()[has_latlon],
        longitudes.ravel()[has_latlon],
        target_lats,
        target_lons,
    )
    return data.reshape(*data.shape[:-2], grid.size)[..., has_latlon[nearest]]


def points_to_grid(
    grid: Grid,
    source_lats: ArrayLike,
    source_lons: ArrayLike,
    values: ArrayLike,
    distance_scale_km: float,
) -> NDArray[np.floating]:
    """Interpolate scattered observations onto a grid with Gaussian weights.

    This wraps autogrid.points.gaussian_weighted_mean, which describes the
    weights. Every observation contributes to every grid point, so grid points
    far from all observations take the values of the nearest ones.

    Args:
        grid: Grid to interpolate onto.
        source_lats: Latitude of each observation in degrees, of any shape.
        source_lons: Longitude of each observation in degrees, of the same shape.
        values: Observed values, whose trailing dimensions match the source shape,
            e.g. (n_variables, n_obs). NaN or infinite observations are ignored.
            Only NaN marks missing values, so fill a masked array with
            values.filled(np.nan) first.
        distance_scale_km: Standard deviation of the Gaussian weights in km.

    Returns:
        An array with the leading dimensions of values followed by the grid shape.
        Grid points with no valid observations, or with no latitude or longitude,
        are NaN.

    Raises:
        ValueError: As for gaussian_weighted_mean.
    """
    flat_longitudes, flat_latitudes = (coords.ravel() for coords in grid.lonlat())
    # Exclude grid points beyond the valid area of a projection
    has_latlon = np.isfinite(flat_latitudes) & np.isfinite(flat_longitudes)
    gridded = gaussian_weighted_mean(
        source_lats,
        source_lons,
        values,
        flat_latitudes[has_latlon],
        flat_longitudes[has_latlon],
        distance_scale_km,
    )
    result = np.full((*gridded.shape[:-1], grid.size), np.nan, dtype=gridded.dtype)
    result[..., has_latlon] = gridded
    return result.reshape(*gridded.shape[:-1], *grid.shape)
