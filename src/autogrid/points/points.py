"""Nearest neighbours and weighted means of points, by great circle distance.

These need only the latitudes and longitudes of the points: no CRS, and no
rectilinear structure.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial import KDTree

from autogrid.points._geometry import chord_to_km, unit_vectors

# Largest number of (target, source) weights held in memory at once by
# gaussian_weighted_mean
_CHUNK_ELEMENTS = 2**24


def nearest_indices(
    source_lats: ArrayLike,
    source_lons: ArrayLike,
    target_lats: ArrayLike,
    target_lons: ArrayLike,
) -> tuple[NDArray[np.intp], ...]:
    """Source indices of the nearest source point to each target point.

    This suits flat point lists, such as an anemoi dataset's latitudes and
    longitudes, and curvilinear grids. It gives the same nearest points as the
    "nearest_great_circle" method of Regridder, which uses it.

    Every target point takes its nearest source point, however far away, so
    nothing is masked outside the source domain.

    Args:
        source_lats: Latitude of each source point in degrees, of any shape.
        source_lons: Longitude of each source point in degrees, of the same shape.
        target_lats: Latitude of each target point in degrees, of any shape.
        target_lons: Longitude of each target point in degrees, of the same shape.

    Returns:
        Tuple with one integer array per source dimension, each with the target
        shape, indexing the nearest source point. Gather data whose trailing
        dimensions match the source shape with data[(..., *indices)]. For a
        1D source this is a 1-tuple, e.g. data[..., indices[0]].

    Raises:
        ValueError: If latitudes and longitudes differ in shape, are empty, or
            are not all finite.
    """
    source_lats, source_lons = check_points("Source", source_lats, source_lons)
    target_lats, target_lons = check_points("Target", target_lats, target_lons)
    # Straight-line distance between points on the unit sphere increases with
    # great circle distance, so a KD-tree in 3D finds the same nearest point
    tree = KDTree(unit_vectors(source_lats, source_lons))
    _, flat_indices = tree.query(unit_vectors(target_lats, target_lons), workers=-1)
    flat_indices = np.asarray(flat_indices, dtype=np.intp).reshape(target_lats.shape)
    return np.unravel_index(flat_indices, source_lats.shape)


def gaussian_weighted_mean(
    source_lats: ArrayLike,
    source_lons: ArrayLike,
    values: ArrayLike,
    target_lats: ArrayLike,
    target_lons: ArrayLike,
    distance_scale_km: float,
) -> NDArray[np.floating]:
    """Interpolate scattered observations onto target points with Gaussian weights.

    Each target point is the weighted mean of every observation, with weight
    exp(-0.5 * (d / distance_scale_km) ** 2) for great circle distance d. This
    suits observations whose positions change from one time to the next, such
    as Argo floats, so no weights are cached.

    Target points far from every observation take the values of the nearest
    ones, since their weights dominate. Weights are scaled relative to the
    nearest observation before exponentiating, so they do not underflow
    however far away the observations are. If the nearest observation is NaN
    for some variable, that variable can still be NaN far from every valid
    observation.

    Every observation is weighted at every target point, which costs time
    proportional to their product. Memory is bounded by computing the weights
    for blocks of target points in turn.

    Args:
        source_lats: Latitude of each observation in degrees, of any shape.
        source_lons: Longitude of each observation in degrees, of the same shape.
        values: Observed values, whose trailing dimensions match the source
            shape, e.g. (n_variables, n_obs). NaN or infinite observations are
            ignored. Only NaN marks missing values, so fill a masked array with
            values.filled(np.nan) first.
        target_lats: Latitude of each target point in degrees, of any shape,
            e.g. Grid.latitudes.
        target_lons: Longitude of each target point in degrees, of the same shape.
        distance_scale_km: Standard deviation of the Gaussian weights in km.

    Returns:
        An array with the leading dimensions of values followed by the target
        shape. Target points with no valid observations are NaN.

    Raises:
        ValueError: If latitudes and longitudes differ in shape, are empty, or
            are not all finite, if values do not end with the source shape, or
            if distance_scale_km is not positive.
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

    leading_shape = values.shape[: values.ndim - len(source_shape)]
    flat_values = values.reshape(-1, source_lats.size).astype(np.float64, copy=False)
    # Drop NaN or infinite observations from both the weighted sum and the sum
    # of weights
    is_valid = np.isfinite(flat_values)
    valid = is_valid.astype(np.float64)
    filled = np.where(is_valid, flat_values, 0.0)

    source_xyz = unit_vectors(source_lats, source_lons)
    target_xyz = unit_vectors(target_lats, target_lons)

    result = np.empty((flat_values.shape[0], len(target_xyz)))
    chunk = max(1, _CHUNK_ELEMENTS // len(source_xyz))
    for start in range(0, len(target_xyz), chunk):
        block = slice(start, start + chunk)
        # |a - b|^2 = 2 - 2 a.b for unit vectors a and b
        chord_sq = 2.0 - 2.0 * (target_xyz[block] @ source_xyz.T)
        distance_km = chord_to_km(np.sqrt(chord_sq.clip(min=0.0)))
        scaled = (distance_km / distance_scale_km) ** 2
        # Subtracting each target point's smallest exponent rescales its weights
        # without changing their weighted mean, and keeps the largest weight at 1
        # so that weights far from every observation do not all underflow to 0
        weights = np.exp(-0.5 * (scaled - scaled.min(axis=1, keepdims=True)))
        with np.errstate(invalid="ignore", divide="ignore"):
            result[:, block] = (filled @ weights.T) / (valid @ weights.T)

    result[~np.isfinite(result)] = np.nan
    result = result.astype(np.result_type(values.dtype, np.float32))
    return result.reshape((*leading_shape, *target_lats.shape))


def check_points(
    name: str, latitudes: ArrayLike, longitudes: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return latitudes and longitudes as float arrays, checking they are valid."""
    latitudes = np.asarray(latitudes, dtype=np.float64)
    longitudes = np.asarray(longitudes, dtype=np.float64)
    if latitudes.shape != longitudes.shape:
        msg = (
            f"{name} latitudes and longitudes must have the same shape, but got "
            f"shapes {latitudes.shape} and {longitudes.shape}."
        )
        raise ValueError(msg)
    if latitudes.size == 0:
        msg = f"{name} points must not be empty."
        raise ValueError(msg)
    if not (np.isfinite(latitudes).all() and np.isfinite(longitudes).all()):
        msg = f"{name} latitudes and longitudes must all be finite."
        raise ValueError(msg)
    return latitudes, longitudes
