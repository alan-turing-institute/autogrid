"""Points, and their nearest neighbours and weighted means by great circle distance.

These need only latitudes and longitudes: no CRS, and no Grid.

- points: Points, for points of any shape known only by their latitudes and longitudes.
- great_circle: nearest neighbours and Gaussian weighted means of lat/lon points.
"""

from autogrid.points.great_circle import gaussian_weighted_mean, nearest_indices
from autogrid.points.points import Points

__all__ = [
    "Points",
    "gaussian_weighted_mean",
    "nearest_indices",
]
