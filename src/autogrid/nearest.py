"""Nearest neighbour search by great circle distance, from latitudes and longitudes."""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial import cKDTree


def _unit_vectors(
    latitudes: NDArray[np.float64], longitudes: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return the (n, 3) unit sphere position of each point, flattened."""
    lat, lon = np.deg2rad(latitudes).ravel(), np.deg2rad(longitudes).ravel()
    return np.stack(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1
    )


def _check_points(
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


def nearest_indices(
    source_lats: ArrayLike,
    source_lons: ArrayLike,
    target_lats: ArrayLike,
    target_lons: ArrayLike,
) -> tuple[NDArray[np.intp], ...]:
    """Source indices of the nearest source point to each target point.

    Points are compared by great circle distance on a sphere, so only their
    latitudes and longitudes are needed: no CRS, and no rectilinear structure.
    This suits flat point lists, such as an anemoi dataset's latitudes and
    longitudes, and curvilinear grids. It gives the same nearest points as the
    "nearest_great_circle" method of Regridder.

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
    source_lats, source_lons = _check_points("Source", source_lats, source_lons)
    target_lats, target_lons = _check_points("Target", target_lats, target_lons)
    # Straight-line distance between points on the unit sphere increases with
    # great circle distance, so a KD-tree in 3D finds the same nearest point
    tree = cKDTree(_unit_vectors(source_lats, source_lons))
    _, flat_indices = tree.query(_unit_vectors(target_lats, target_lons), workers=-1)
    flat_indices = np.asarray(flat_indices, dtype=np.intp).reshape(target_lats.shape)
    return np.unravel_index(flat_indices, source_lats.shape)
