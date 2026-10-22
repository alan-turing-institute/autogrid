"""Nearest neighbours and weighted means of points, by great circle distance.

These need only the latitudes and longitudes of the points: no CRS, and no
rectilinear structure.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial import KDTree

from autogrid.points._geometry import chord_to_km, unit_vectors
from autogrid.points.points import Points

# Largest number of (target, source) pairs in each block of gaussian_weighted_mean.
# Each block holds several float64 arrays of this size at once, about 1 GB in all.
_CHUNK_ELEMENTS = 2**24


def gaussian_weighted_mean(
    *,
    source_lats: ArrayLike,
    source_lons: ArrayLike,
    data: ArrayLike,
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
        data: Observed values, whose trailing dimensions match the source
            shape, e.g. (n_variables, n_obs). NaN or infinite observations are
            ignored. Only NaN marks missing values, so fill a masked array with
            data.filled(np.nan) first.
        target_lats: Latitude of each target point in degrees, of any shape,
            e.g. from Grid.latlon().
        target_lons: Longitude of each target point in degrees, of the same shape.
        distance_scale_km: Standard deviation of the Gaussian weights in km.

    Returns:
        An array with the leading dimensions of data followed by the target
        shape. Target points with no valid observations are NaN.

    Raises:
        ValueError: If latitudes and longitudes differ in shape, are empty, or
            are not all finite, if data does not end with the source shape, or
            if distance_scale_km is not positive.
    """
    source = Points(source_lats, source_lons, name="source")
    target = Points(target_lats, target_lons, name="target")
    data = np.asarray(data)
    if data.shape[data.ndim - len(source.shape) :] != source.shape:
        msg = (
            f"Data shape {data.shape} does not end with the source points "
            f"shape {source.shape}."
        )
        raise ValueError(msg)
    if not distance_scale_km > 0:
        msg = f"distance_scale_km must be positive, but got {distance_scale_km}."
        raise ValueError(msg)

    leading_shape = data.shape[: data.ndim - len(source.shape)]
    flat_data = data.reshape(-1, source.size).astype(np.float64, copy=False)
    # Drop NaN or infinite observations from both the weighted sum and the sum
    # of weights
    is_valid = np.isfinite(flat_data)
    valid = is_valid.astype(np.float64)
    filled = np.where(is_valid, flat_data, 0.0)

    source_xyz = unit_vectors(source.latitudes, source.longitudes)
    target_xyz = unit_vectors(target.latitudes, target.longitudes)

    result = np.empty((flat_data.shape[0], len(target_xyz)))
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
    result = result.astype(np.result_type(data.dtype, np.float32))
    return result.reshape((*leading_shape, *target.shape))


def nearest_indices(
    *,
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
    source = Points(source_lats, source_lons, name="source")
    target = Points(target_lats, target_lons, name="target")
    # Straight-line distance between points on the unit sphere increases with
    # great circle distance, so a KD-tree in 3D finds the same nearest point
    tree = KDTree(unit_vectors(source.latitudes, source.longitudes))
    _, flat_indices = tree.query(
        unit_vectors(target.latitudes, target.longitudes), workers=-1
    )
    flat_indices = np.asarray(flat_indices, dtype=np.intp).reshape(target.shape)
    return np.unravel_index(flat_indices, source.shape)
