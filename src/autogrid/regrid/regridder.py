"""Regrid numpy arrays between Grids, with iris or by great circle distance."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from autogrid.grids import Grid
from autogrid.regrid.methods import _METHODS, Method, RegridMethod


class Regridder:
    """Regrid arrays from one Grid to another, reusing the regridding weights.

    Build one Regridder per (source, target) pair and apply it to each field,
    e.g. each time step of a forecast. The weights for each method are built
    the first time that method is used and cached for later calls.
    """

    def __init__(self, source: Grid, target: Grid) -> None:
        """Initialise a Regridder.

        Args:
            source: Grid the input arrays are on.
            target: Grid to regrid onto.
        """
        self.source = source
        self.target = target
        self._methods: dict[Method, RegridMethod] = {}

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.source!r} -> {self.target!r})"

    def _method(self, method: Method) -> RegridMethod:
        """Return the cached regridding method, building it if needed."""
        if method not in self._methods:
            if method not in _METHODS:
                msg = f"Unknown regridding method {method!r}."
                raise ValueError(msg)
            self._methods[method] = _METHODS[method](self.source, self.target)
        return self._methods[method]

    def outside(self, method: Method = "linear") -> NDArray[np.bool_]:
        """Flag for target points that lie outside the source grid for a method.

        These are the target points that the method leaves NaN for a source
        field with no NaN values. The iris methods find them in the source CRS
        after any datum shift from the target, while "nearest_great_circle"
        finds them from where its search starts.

        Returns:
            A boolean array with the target grid shape (ny, nx).

        Raises:
            ValueError: If the method is unknown.
        """
        return self._method(method).outside

    def apply(self, data: ArrayLike, method: Method = "linear") -> NDArray[np.floating]:
        """Regrid an array with shape (..., ny, nx) on the source grid.

        Args:
            data: Array whose last two dimensions match the source grid shape.
            method: "linear", "nearest", "nearest_great_circle" or
                "area_weighted". "nearest" finds the nearest point in the source
                grid's own coordinates, which for a latitude-longitude source
                means nearest in degrees, while "nearest_great_circle" finds the
                nearest point along the Earth's surface. Area weighted
                regridding requires both grids to share a coordinate system.

        Returns:
            An array with shape (..., ny, nx) on the target grid. Target points
            outside the source grid, or derived from NaN or masked source
            points, are NaN.

        Raises:
            ValueError: If the data shape does not end with the source grid
                shape, or the method is unknown.
        """
        data = np.ma.masked_invalid(data)
        if data.shape[-2:] != self.source.shape:
            msg = (
                f"Data shape {data.shape} does not end with the source grid "
                f"shape {self.source.shape}."
            )
            raise ValueError(msg)
        result = self._method(method)(data)
        return np.ma.filled(result.astype(np.result_type(result, np.float32)), np.nan)
