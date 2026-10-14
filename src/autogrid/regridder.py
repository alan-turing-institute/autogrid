"""Regrid numpy arrays between Grids, using iris internally."""

from collections.abc import Callable
from functools import cached_property
from typing import Literal

import numpy as np
from iris.analysis import AreaWeighted, Linear, Nearest
from iris.cube import Cube
from numpy.typing import ArrayLike, NDArray

from autogrid.grid import Grid

Method = Literal["linear", "nearest", "area_weighted"]


def _cube(grid: Grid, data: ArrayLike | None = None) -> Cube:
    """Build an iris cube on a Grid, with any leading dimensions of data."""
    shape = grid.shape if data is None else np.shape(data)
    n_lead = len(shape) - 2
    return Cube(
        data,
        shape=None if data is not None else shape,
        dim_coords_and_dims=[
            (grid.y_coord.copy(), n_lead),
            (grid.x_coord.copy(), n_lead + 1),
        ],
    )


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
        self._regridders: dict[Method, Callable[[Cube], Cube]] = {}

    @cached_property
    def _outside(self) -> NDArray[np.bool_]:
        """Flag target points that lie outside the cells of the source grid."""
        source, target = self.source, self.target
        xy = source.crs.transform_points(target.crs, target.x, target.y)
        x, y = xy[..., 0], xy[..., 1]
        x_lo, x_hi = source.x_coord.bounds.min(), source.x_coord.bounds.max()
        y_lo, y_hi = source.y_coord.bounds.min(), source.y_coord.bounds.max()
        outside = ~np.isfinite(x) | ~np.isfinite(y) | (y < y_lo) | (y > y_hi)
        if source.x_coord.circular:
            return outside
        if source.x_coord.units == "degrees":
            x = x_lo + (x - x_lo) % 360
        return outside | (x < x_lo) | (x > x_hi)

    def _regridder(self, method: Method) -> Callable[[Cube], Cube]:
        """Return the cached iris regridder for a method, building it if needed."""
        if method not in self._regridders:
            match method:
                # Extrapolation only fills the half cell between the outermost
                # source points and the source grid edge: anything further out
                # is masked by _outside
                case "linear":
                    scheme = Linear(extrapolation_mode="extrapolate")
                case "nearest":
                    scheme = Nearest(extrapolation_mode="extrapolate")
                case "area_weighted":
                    scheme = AreaWeighted()
                case _:
                    msg = f"Unknown regridding method {method!r}."
                    raise ValueError(msg)
            self._regridders[method] = scheme.regridder(
                _cube(self.source), _cube(self.target)
            )
        return self._regridders[method]

    def apply(self, data: ArrayLike, method: Method = "linear") -> NDArray[np.floating]:
        """Regrid an array with shape (..., ny, nx) on the source grid.

        Args:
            data: Array whose last two dimensions match the source grid shape.
            method: "linear", "nearest" or "area_weighted". Area weighted
                regridding requires both grids to share a coordinate system.

        Returns:
            An array with shape (..., ny, nx) on the target grid. Target points
            outside the source grid, or derived from NaN or masked source
            points, are NaN.
        """
        data = np.ma.masked_invalid(data)
        if data.shape[-2:] != self.source.shape:
            msg = (
                f"Data shape {data.shape} does not end with the source grid "
                f"shape {self.source.shape}."
            )
            raise ValueError(msg)
        result = self._regridder(method)(_cube(self.source, data)).data
        result = np.ma.filled(result.astype(np.result_type(result, np.float32)), np.nan)
        result[..., self._outside] = np.nan
        return result

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.source!r} -> {self.target!r})"
