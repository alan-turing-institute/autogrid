"""Regrid numpy arrays between Grids with sparse weight matrices."""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse

from autogrid.grids import Grid
from autogrid.regrid.methods import METHODS, Method, WeightsFunction


class Regridder:
    """Regrid arrays from one Grid to another with one method, reusing the weights.

    Build one Regridder per (source, target, method) and apply it to each field. The
    weights are built when the Regridder is created.

    Attributes:
        source: Grid the input arrays are on.
        target: Grid the output arrays are on.
        method: Function that built the weights.
        weights: Sparse matrix with shape (target.size, source.size) giving the weight
            of each source point in each target point, with no explicit zeros.
    """

    def __init__(
        self, source: Grid, target: Grid, method: Method | WeightsFunction = "linear"
    ) -> None:
        """Initialise a Regridder and build its weights.

        Args:
            source: Grid the input arrays are on.
            target: Grid to regrid onto.
            method: The name of a method in autogrid.regrid.methods, as listed by
                Method, or a function building the weights from the source and target
                grids, following the rules in autogrid.regrid.methods.

        Raises:
            ValueError: If the method is unknown, its weights have the wrong shape, or
                the method rejects the grids, e.g. area weighting grids that do not
                share a coordinate system.
        """
        self.source = source
        self.target = target
        # Determine the regridding method
        if not callable(method):
            if method not in METHODS:
                msg = f"Unknown regridding method {method!r}."
                raise ValueError(msg)
            method = METHODS[method]
        self.method = method
        # Construct the weight matrix
        weights = sparse.csr_array(method(source, target))
        if weights.shape != (target.size, source.size):
            msg = (
                f"Regridding weights have shape {weights.shape}, but should have shape "
                f"{(target.size, source.size)}."
            )
            raise ValueError(msg)
        weights.eliminate_zeros()
        self.weights = weights
        # Target points with no contributing source points
        self._empty = np.diff(weights.indptr) == 0

    def __repr__(self) -> str:
        """Return a string representation of the Regridder."""
        name = getattr(self.method, "__name__", repr(self.method))
        return f"{type(self).__name__}({self.source!r} -> {self.target!r}, {name})"

    def apply(self, data: ArrayLike) -> NDArray[np.floating]:
        """Regrid an array with shape (..., ny, nx) on the source grid.

        Args:
            data: Array whose last two dimensions match the source grid shape.

        Returns:
            An array with shape (..., ny, nx) on the target grid. Target points outside
            the source grid, or derived from NaN or infinite source points, are NaN.
            Only NaN marks missing values, so fill a masked array with
            data.filled(np.nan) first.

        Raises:
            ValueError: If the data shape does not end with the source grid shape.
        """
        data = np.asarray(data)
        if data.shape[-2:] != self.source.shape:
            msg = (
                f"Data shape {data.shape} does not end with the source grid shape "
                f"{self.source.shape}."
            )
            raise ValueError(msg)
        values = data.reshape(-1, self.source.size)
        # Mark NaN and infinite points as invalid
        invalid = ~np.isfinite(values)
        # Zero invalid points before weighting them
        result = (self.weights @ np.where(invalid, 0, values).T).T
        # Set result to NaN where any invalid source points contributed to it
        if invalid.any():
            result[(self.weights @ invalid.T.astype(np.float32)).T > 0] = np.nan
        # Set result to NaN for any target points with no contributing source points
        result[:, self._empty] = np.nan
        dtype = np.result_type(data.dtype, np.float32)
        return result.astype(dtype).reshape(*data.shape[:-2], *self.target.shape)
