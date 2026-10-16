"""Regrid numpy arrays between Grids, with iris or by great circle distance."""

from collections.abc import Callable
from functools import cached_property
from typing import Literal

import numpy as np
from iris.analysis import AreaWeighted, Linear, Nearest
from iris.coords import DimCoord
from iris.cube import Cube
from numpy.typing import ArrayLike, NDArray

from autogrid.adaptors.iris import sample_transformer, to_cube
from autogrid.grids import Grid
from autogrid.regrid import grid_search

Method = Literal["linear", "nearest", "nearest_great_circle", "area_weighted"]


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

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.source!r} -> {self.target!r})"

    @cached_property
    def _source_coords(self) -> tuple[DimCoord, ...]:
        """Iris (y, x) coordinates of the source grid, shared by every source cube."""
        return to_cube(self.source).dim_coords

    @cached_property
    def _great_circle(
        self,
    ) -> tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.bool_]]:
        """Source (iy, ix) of the great circle nearest source point to each target,
        and the flag for targets outside the source grid, from the same search."""
        return grid_search.search_grid(self.source, self.target)

    @cached_property
    def outside_source(self) -> NDArray[np.bool_]:
        """Flag for target points that lie outside the cells of the source grid.

        Target points are transformed into the source CRS as iris does for the
        "linear", "nearest" and "area_weighted" methods, applying any datum
        shift between the grids. "nearest_great_circle" instead masks the
        points whose search starts outside the source grid, found through each
        grid's own latitudes and longitudes with no datum shift, so it can
        differ from this near the edge of the source grid if the datums differ.

        Returns:
            A boolean array with the target grid shape (ny, nx).
        """
        return grid_search.outside_grid(
            self.source,
            self.target,
            sample_transformer(self.source.crs, self.target.crs),
        )

    def _regridder(self, method: Method) -> Callable[[Cube], Cube]:
        """Return the cached iris regridder for a method, building it if needed."""
        if method not in self._regridders:
            match method:
                # Extrapolation only fills the half cell between the outermost
                # source points and the source grid edge: anything further out
                # is masked by `outside_source`.
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
                to_cube(self.source, coords=self._source_coords), to_cube(self.target)
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
        if method == "nearest_great_circle":
            iy, ix, outside = self._great_circle
            result = data[..., iy, ix]
        else:
            result = self._regridder(method)(
                to_cube(self.source, data, self._source_coords)
            ).data
            outside = self.outside_source
        result = np.ma.filled(result.astype(np.result_type(result, np.float32)), np.nan)
        result[..., outside] = np.nan
        return result
