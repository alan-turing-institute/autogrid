"""Points on a spherical Earth, from latitudes and longitudes."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

# Mean Earth radius in km, as used by the haversine package
EARTH_RADIUS_KM = 6371.0088


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


def chord_to_km(chord: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert straight-line distances between unit vectors to great circle km."""
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.clip(chord, 0.0, 2.0) / 2.0)


def km_to_chord(distance_km: ArrayLike) -> NDArray[np.float64]:
    """Convert great circle km to straight-line distances between unit vectors."""
    angle = np.clip(
        np.asarray(distance_km, dtype=np.float64) / EARTH_RADIUS_KM, 0, np.pi
    )
    return 2.0 * np.sin(angle / 2.0)


def unit_vectors(
    latitudes: NDArray[np.float64], longitudes: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return the (n, 3) unit sphere position of each point, flattened."""
    lat, lon = np.deg2rad(latitudes).ravel(), np.deg2rad(longitudes).ravel()
    return np.stack(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1
    )
