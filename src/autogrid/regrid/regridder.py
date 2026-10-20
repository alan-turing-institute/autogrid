"""Regrid numpy arrays between Grids with sparse weight matrices."""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse

from autogrid.grids import Grid
from autogrid.regrid.methods import _METHODS, Method


class Regridder:
    """Regrid arrays from one Grid to another, reusing the regridding weights.

    Build one Regridder per (source, target) pair and apply it to each field. The
    weights for each method are built the first time that method is used and cached for
    later calls.
    """

    def __init__(self, source: Grid, target: Grid) -> None:
        """Initialise a Regridder.

        Args:
            source: Grid the input arrays are on.
            target: Grid to regrid onto.
        """
        self.source = source
        self.target = target
        self._weights: dict[Method, sparse.csr_array] = {}

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.source!r} -> {self.target!r})"

    def apply(self, data: ArrayLike, method: Method = "linear") -> NDArray[np.floating]:
        """Regrid an array with shape (..., ny, nx) on the source grid.

        Args:
            data: Array whose last two dimensions match the source grid shape.
            method: "linear", "nearest", "nearest_great_circle" or "area_weighted".
                "nearest" finds the nearest point in the source grid's own coordinates,
                which for a latitude-longitude source means nearest in degrees, while
                "nearest_great_circle" finds the nearest point along the Earth's
                surface. Area weighted regridding requires both grids to share a
                coordinate system.

        Returns:
            An array with shape (..., ny, nx) on the target grid. Target points outside
            the source grid, or derived from NaN or masked source points, are NaN.

        Raises:
            ValueError: If the data shape does not end with the source grid shape, the
                method is unknown, or the grids do not share a coordinate system for
                area weighting.
        """
        data = np.asanyarray(data)
        if data.shape[-2:] != self.source.shape:
            msg = (
                f"Data shape {data.shape} does not end with the source grid shape "
                f"{self.source.shape}."
            )
            raise ValueError(msg)
        weights = self.weights(method)
        values = np.ma.getdata(data).reshape(-1, self.source.size)
        # Mark masked, NaN and infinite points as invalid
        invalid = np.ma.getmaskarray(data).reshape(-1, self.source.size)
        invalid = invalid | ~np.isfinite(values)
        # Zero invalid points before weighting them
        result = (weights @ np.where(invalid, 0, values).T).T
        # Set result to NaN where any invalid source points contributed to it
        if invalid.any():
            result[(weights @ invalid.T.astype(np.float32)).T > 0] = np.nan
        # Set result to NaN for any target points with no contributing source points
        result[:, np.diff(weights.indptr) == 0] = np.nan
        dtype = np.result_type(data.dtype, np.float32)
        return result.astype(dtype).reshape(*data.shape[:-2], *self.target.shape)

    def weights(self, method: Method) -> sparse.csr_array:
        """Return the cached weights of a method, building them if needed."""
        if method not in self._weights:
            if method not in _METHODS:
                msg = f"Unknown regridding method {method!r}."
                raise ValueError(msg)
            self._weights[method] = _METHODS[method](self.source, self.target)
        return self._weights[method]
