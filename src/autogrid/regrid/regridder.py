"""Regrid numpy arrays between Grids, with iris or by great circle distance."""

from collections.abc import Callable
from functools import cached_property
from typing import Literal

import numpy as np
from iris.analysis import AreaWeighted, Linear, Nearest
from numpy.typing import ArrayLike, NDArray

from autogrid.grids import Grid
from autogrid.regrid import grid_search
from autogrid.regrid.cubes import pad_edges, padded_coords, to_cube

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
        self._regridders: dict[
            Method, Callable[[np.ma.MaskedArray], np.ma.MaskedArray]
        ] = {}

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.source!r} -> {self.target!r})"

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

        These are the target points that the "linear" method leaves NaN for a
        source field with no NaN values. They are found by iris, so match the
        "linear" and "nearest" methods.

        Returns:
            A boolean array with the target grid shape (ny, nx).
        """
        return np.isnan(self.apply(np.ones(self.source.shape, np.float32)))

    def _regridder(
        self, method: Method
    ) -> Callable[[np.ma.MaskedArray], np.ma.MaskedArray]:
        """Return the cached iris regridding function for a method, building it if
        needed."""
        if method not in self._regridders:
            # The source is padded to its cell edges, so masking beyond the
            # outermost source points masks targets outside the source cells
            coords, pad = padded_coords(self.source)
            match method:
                case "linear":
                    scheme = Linear(extrapolation_mode="mask")
                case "nearest":
                    scheme = Nearest(extrapolation_mode="mask")
                case "area_weighted":
                    # Area weighting uses the source cells, so is not padded
                    scheme, coords, pad = (
                        AreaWeighted(),
                        to_cube(self.source).dim_coords,
                        None,
                    )
                case _:
                    msg = f"Unknown regridding method {method!r}."
                    raise ValueError(msg)
            regridder = scheme.regridder(
                to_cube(self.source, coords=coords), to_cube(self.target)
            )

            def regrid(data: np.ma.MaskedArray) -> np.ma.MaskedArray:
                if pad is not None:
                    data = pad_edges(data, pad)
                return regridder(to_cube(self.source, data, coords)).data

            self._regridders[method] = regrid
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
            result[..., outside] = np.ma.masked
        else:
            result = self._regridder(method)(data)
        return np.ma.filled(result.astype(np.result_type(result, np.float32)), np.nan)
