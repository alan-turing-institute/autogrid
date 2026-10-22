"""Locations: anything with a latitude and longitude for each of its points."""

from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class Locations(Protocol):
    """Anything with the latitude and longitude of each of its points.

    A Grid is Locations, and so are Points, e.g. observation sites, a flat list of
    points or the 2D points of a curvilinear grid.
    """

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of the points, which arrays on them end with."""
        ...

    @property
    def size(self) -> int:
        """The number of points."""
        ...

    def latlon(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """WGS 84 latitude and longitude of the points, each with their shape.

        Points without a latitude or longitude, e.g. beyond the valid area of a
        projection, are not finite.
        """
        ...
