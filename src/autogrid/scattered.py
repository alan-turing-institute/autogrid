"""Grid scattered point observations, weighting them by great circle distance."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from autogrid.grid import Grid
from autogrid.sphere import check_points, chord_to_km, unit_vectors

# Largest number of (target, source) weights held in memory at once
_CHUNK_ELEMENTS = 2**24


def points_to_grid(
    source_lats: ArrayLike,
    source_lons: ArrayLike,
    values: ArrayLike,
    target_grid: Grid,
    distance_scale_km: float,
    min_weight: float = 1e-10,
) -> NDArray[np.floating]:
    """Interpolate scattered observations onto a Grid with Gaussian weights."""
    return points_to_points(
        source_lats,
        source_lons,
        values,
        target_grid.latitudes,
        target_grid.longitudes,
        distance_scale_km,
        min_weight,
    )


def points_to_points(
    source_lats: ArrayLike,
    source_lons: ArrayLike,
    values: ArrayLike,
    target_lats: ArrayLike,
    target_lons: ArrayLike,
    distance_scale_km: float,
    min_weight: float = 1e-10,
) -> NDArray[np.floating]:
    """Interpolate scattered observations onto target points with Gaussian weights.

    Each target point is the weighted mean of every observation, with weight
    exp(-0.5 * (d / distance_scale_km) ** 2) for great circle distance d,
    floored at min_weight. This suits observations whose positions change from
    one time to the next, such as Argo floats, so no weights are cached.

    With min_weight > 0, target points far from every observation tend to the
    unweighted mean of the observations rather than to NaN. Set min_weight to 0
    to leave such points NaN once their weights underflow.

    Args:
        source_lats: Latitude of each observation in degrees, of any shape.
        source_lons: Longitude of each observation in degrees, of the same shape.
        values: Observed values, whose trailing dimensions match the source
            shape, e.g. (n_variables, n_obs). NaN observations are ignored.
        target_lats: Latitude of each target point in degrees, of any shape,
            e.g. Grid.latitudes.
        target_lons: Longitude of each target point in degrees, of the same shape.
        distance_scale_km: Standard deviation of the Gaussian weights in km.
        min_weight: Smallest weight given to any observation.

    Returns:
        An array with the leading dimensions of values followed by the target
        shape. Target points with no valid observations are NaN.

    Raises:
        ValueError: If latitudes and longitudes differ in shape, are empty, or
            are not all finite, if values do not end with the source shape, or
            if distance_scale_km is not positive or min_weight is negative.
    """
    source_lats, source_lons = check_points("Source", source_lats, source_lons)
    target_lats, target_lons = check_points("Target", target_lats, target_lons)
    values = np.asarray(values)
    source_shape = source_lats.shape
    if values.shape[values.ndim - len(source_shape) :] != source_shape:
        msg = (
            f"Values shape {values.shape} does not end with the source points "
            f"shape {source_shape}."
        )
        raise ValueError(msg)
    if not distance_scale_km > 0:
        msg = f"distance_scale_km must be positive, but got {distance_scale_km}."
        raise ValueError(msg)
    if not min_weight >= 0:
        msg = f"min_weight must not be negative, but got {min_weight}."
        raise ValueError(msg)

    leading_shape = values.shape[: values.ndim - len(source_shape)]
    flat_values = values.reshape(-1, source_lats.size).astype(np.float64)
    # Drop NaN observations from both the weighted sum and the sum of weights
    valid = np.isfinite(flat_values)
    filled = np.where(valid, flat_values, 0.0)
    valid = valid.astype(np.float64)

    source_xyz = unit_vectors(source_lats, source_lons)
    target_xyz = unit_vectors(target_lats, target_lons)
    result = np.empty((flat_values.shape[0], len(target_xyz)))
    chunk = max(1, _CHUNK_ELEMENTS // len(source_xyz))
    for start in range(0, len(target_xyz), chunk):
        block = slice(start, start + chunk)
        # |a - b|^2 = 2 - 2 a.b for unit vectors a and b
        chord_sq = 2.0 - 2.0 * (target_xyz[block] @ source_xyz.T)
        distance_km = chord_to_km(np.sqrt(chord_sq.clip(min=0.0)))
        weights = np.exp(-0.5 * (distance_km / distance_scale_km) ** 2)
        weights = weights.clip(min=min_weight)  # shape: (n_block, n_source)
        with np.errstate(invalid="ignore", divide="ignore"):
            result[:, block] = (filled @ weights.T) / (valid @ weights.T)

    result[~np.isfinite(result)] = np.nan
    result = result.astype(np.result_type(values.dtype, np.float32))
    return result.reshape(*leading_shape, *target_lats.shape)
