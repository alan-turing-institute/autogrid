"""Regrid numpy arrays between Grids, using iris internally."""

from collections.abc import Callable
from functools import cached_property
from typing import Literal, get_args

import numpy as np
from iris.analysis import AreaWeighted, Linear, Nearest
from iris.cube import Cube
from numpy.typing import ArrayLike, NDArray
from pyproj import Transformer

from autogrid import nearest
from autogrid.adaptors.iris import to_cube
from autogrid.grid import Grid

NearestMethod = Literal["nearest", "nearest_great_circle"]
Method = Literal["linear", "area_weighted"] | NearestMethod


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
        self._nearest_indices: dict[
            NearestMethod, tuple[NDArray[np.intp], NDArray[np.intp]]
        ] = {}

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.source!r} -> {self.target!r})"

    @cached_property
    def outside_source(self) -> NDArray[np.bool_]:
        """Flag target points that lie outside the cells of the source grid.

        Returns:
            A boolean array with the target grid shape (ny, nx).
        """
        source, target = self.source, self.target
        x, y = Transformer.from_crs(target.crs, source.crs, always_xy=True).transform(
            target.x, target.y
        )
        x_lo, x_hi = source.x_bounds.min(), source.x_bounds.max()
        y_lo, y_hi = source.y_bounds.min(), source.y_bounds.max()
        outside = ~np.isfinite(x) | ~np.isfinite(y) | (y < y_lo) | (y > y_hi)
        if source.is_circular:
            return outside
        if source.is_geographic:
            x = x_lo + (x - x_lo) % 360
        return outside | (x < x_lo) | (x > x_hi)

    def _regridder(self, method: Method) -> Callable[[Cube], Cube]:
        """Return the cached iris regridder for a method, building it if needed."""
        if method not in self._regridders:
            match method:
                # Extrapolation only fills the half cell between the outermost
                # source points and the source grid edge: anything further out
                # is masked by outside_source
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
                to_cube(self.source), to_cube(self.target)
            )
        return self._regridders[method]

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
        """
        data = np.ma.masked_invalid(data)
        if data.shape[-2:] != self.source.shape:
            msg = (
                f"Data shape {data.shape} does not end with the source grid "
                f"shape {self.source.shape}."
            )
            raise ValueError(msg)
        if method == "nearest_great_circle":
            iy, ix = self.nearest_indices(method)
            result = data[..., iy, ix]
        else:
            result = self._regridder(method)(to_cube(self.source, data)).data
        result = np.ma.filled(result.astype(np.result_type(result, np.float32)), np.nan)
        result[..., self.outside_source] = np.nan
        return result

    def nearest_indices(
        self, method: NearestMethod = "nearest"
    ) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
        """Source grid indices of the nearest source point to each target point.

        Use these to apply nearest neighbour regridding externally, e.g. to
        torch tensors on a GPU, with data[..., iy, ix]. This gives the same
        values as apply(data, method), except that target points outside the
        source grid take the value of the nearest source point at its edge
        rather than NaN. Use outside_source to mask these if needed.

        Args:
            method: "nearest" or "nearest_great_circle", as for apply.

        Returns:
            Tuple of (iy, ix) integer arrays with the target grid shape (ny, nx),
            giving the row and column of the nearest source point.
        """
        if method not in get_args(NearestMethod):
            msg = f"Unknown nearest neighbour method {method!r}."
            raise ValueError(msg)
        if method == "nearest_great_circle" and method not in self._nearest_indices:
            # Search the Grids' own latitudes and longitudes, rather than
            # rebuilding Grids from iris cubes, which can lose CRS detail
            iy, ix = nearest.nearest_indices(
                self.source.latitudes,
                self.source.longitudes,
                self.target.latitudes,
                self.target.longitudes,
            )
            self._nearest_indices[method] = (iy, ix)
        if method not in self._nearest_indices:
            # Regrid the flat index of each source point, which nearest neighbour
            # regridding copies exactly into each target point
            flat = np.arange(np.prod(self.source.shape), dtype=np.float64)
            flat = flat.reshape(self.source.shape)
            result = self._regridder(method)(to_cube(self.source, flat)).data
            iy, ix = np.unravel_index(
                np.asarray(result).astype(np.intp), self.source.shape
            )
            self._nearest_indices[method] = (iy, ix)
        return self._nearest_indices[method]
