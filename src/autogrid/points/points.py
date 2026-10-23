"""Points known only by their latitudes and longitudes."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


class Points:
    """Points of any shape, known only by their latitudes and longitudes.

    Attributes:
        name: Optional label for display.
    """

    def __init__(
        self, latitudes: ArrayLike, longitudes: ArrayLike, *, name: str | None = None
    ) -> None:
        """Initialise Points from the latitude and longitude of each point.

        Args:
            latitudes: Latitude of each point in degrees, of any shape.
            longitudes: Longitude of each point in degrees, of the same shape.
            name: Optional label for display, e.g. "stations". It is not part of the
                points' identity, so is ignored by ==.

        Raises:
            ValueError: If latitudes and longitudes differ in shape, are empty, or are
                not all finite.
        """
        self._validate_latlon(latitudes, longitudes)
        self.name = name

    def __eq__(self, other: object) -> bool:
        """Whether two Points have the same latitudes and longitudes."""
        if not isinstance(other, Points):
            return NotImplemented
        return np.array_equal(self._latitudes, other._latitudes) and np.array_equal(
            self._longitudes, other._longitudes
        )

    def __repr__(self) -> str:
        """Return a string representation of the Points."""
        name = f"name={self.name!r}, " if self.name is not None else ""
        return f"{type(self).__name__}({name}shape={self.shape})"

    @property
    def latitudes(self) -> NDArray[np.float64]:
        """Latitude of each point in degrees, with the shape of the points."""
        return self._latitudes

    @property
    def longitudes(self) -> NDArray[np.float64]:
        """Longitude of each point in degrees, with the shape of the points."""
        return self._longitudes

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of the points."""
        return self._latitudes.shape

    @property
    def size(self) -> int:
        """The number of points."""
        return self._latitudes.size

    def latlon(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Latitude and longitude of the points.

        Returns:
            Tuple of read-only (latitude, longitude) arrays in degrees, with the shape
            of the points.
        """
        return self._latitudes, self._longitudes

    def _validate_latlon(self, latitudes: ArrayLike, longitudes: ArrayLike) -> None:
        """Set latitudes and longitudes as float arrays, checking they are valid.

        Args:
            latitudes: Latitude of each point in degrees, of any shape.
            longitudes: Longitude of each point in degrees, of the same shape.

        Raises:
            ValueError: If latitudes and longitudes differ in shape, are empty, or are
                not all finite.
        """
        # Copy, so that making them read-only leaves the caller's arrays writeable
        latitudes = np.array(latitudes, dtype=np.float64)
        longitudes = np.array(longitudes, dtype=np.float64)
        if latitudes.shape != longitudes.shape:
            msg = (
                "Latitudes and longitudes must have the same shape, but got shapes "
                f"{latitudes.shape} and {longitudes.shape}."
            )
            raise ValueError(msg)
        if latitudes.size == 0:
            msg = "Points must not be empty."
            raise ValueError(msg)
        if not (np.isfinite(latitudes).all() and np.isfinite(longitudes).all()):
            msg = "Latitudes and longitudes must all be finite."
            raise ValueError(msg)
        self._latitudes, self._longitudes = latitudes, longitudes
        self._latitudes.flags.writeable = False
        self._longitudes.flags.writeable = False
