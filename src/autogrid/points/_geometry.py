"""Points on a spherical Earth, from latitudes and longitudes."""

import numpy as np
from numpy.typing import NDArray

# Mean Earth radius in km
EARTH_RADIUS_KM = 6371.0088


def chord_to_km(chord: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert straight-line distances between unit vectors to great circle km."""
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.clip(chord, 0.0, 2.0) / 2.0)


def unit_vectors(
    latitudes: NDArray[np.float64], longitudes: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return the (n, 3) unit sphere position of each point, flattened."""
    lat, lon = np.deg2rad(latitudes).ravel(), np.deg2rad(longitudes).ravel()
    return np.stack(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1
    )
