"""Nearest neighbour search by great circle distance, from latitudes and longitudes."""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial import cKDTree

from autogrid.sphere import check_points, unit_vectors


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
    source_lats, source_lons = check_points("Source", source_lats, source_lons)
    target_lats, target_lons = check_points("Target", target_lats, target_lons)
    # Straight-line distance between points on the unit sphere increases with
    # great circle distance, so a KD-tree in 3D finds the same nearest point
    tree = cKDTree(unit_vectors(source_lats, source_lons))
    _, flat_indices = tree.query(unit_vectors(target_lats, target_lons), workers=-1)
    flat_indices = np.asarray(flat_indices, dtype=np.intp).reshape(target_lats.shape)
    return np.unravel_index(flat_indices, source_lats.shape)
