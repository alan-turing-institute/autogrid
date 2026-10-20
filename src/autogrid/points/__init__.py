"""Nearest neighbours and weighted means of lat/lon points, by great circle distance.

These need only latitudes and longitudes: no CRS, and no Grid.

- points: nearest neighbours and Gaussian weighted means of lat/lon points.
"""

from autogrid.points.points import gaussian_weighted_mean, nearest_indices

__all__ = [
    "gaussian_weighted_mean",
    "nearest_indices",
]
