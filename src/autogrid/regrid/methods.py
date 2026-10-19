"""Regridding methods, each built once for a (source, target) pair of Grids.

To add a method, implement RegridMethod and add it to Method and _METHODS.
"""

from collections.abc import Callable
from functools import cached_property, partial
from typing import Literal, Protocol

import numpy as np
from iris.analysis import AreaWeighted, Linear, Nearest
from numpy.typing import NDArray

from autogrid.grids import Grid
from autogrid.regrid import grid_search
from autogrid.regrid.cubes import pad_edges, padded_coords, to_cube

Method = Literal["linear", "nearest", "nearest_great_circle", "area_weighted"]


class RegridMethod(Protocol):
    """A regridding method, built once for a (source, target) pair of Grids."""

    def __call__(self, data: np.ma.MaskedArray) -> np.ma.MaskedArray:
        """Regrid masked data with shape (..., ny, nx) on the source grid, masking
        target points outside the source grid or derived from masked points."""
        ...

    @property
    def outside(self) -> NDArray[np.bool_]:
        """Flag for target points outside the source grid, with shape (ny, nx)."""
        ...


class IrisMethod(RegridMethod):
    """Regrid with an iris regridding scheme.

    With pad, the source is padded to its cell edges, so a scheme that masks
    beyond the outermost source points masks targets outside the source cells.
    """

    def __init__(
        self,
        scheme: Linear | Nearest | AreaWeighted,
        source: Grid,
        target: Grid,
        *,
        pad: bool = True,
    ) -> None:
        self.source = source
        if pad:
            self._coords, self._pad = padded_coords(source)
        else:
            self._coords, self._pad = to_cube(source).dim_coords, None
        self._regridder = scheme.regridder(
            to_cube(source, coords=self._coords), to_cube(target)
        )

    def __call__(self, data: np.ma.MaskedArray) -> np.ma.MaskedArray:
        if self._pad is not None:
            data = pad_edges(data, self._pad)
        return self._regridder(to_cube(self.source, data, self._coords)).data

    @cached_property
    def outside(self) -> NDArray[np.bool_]:
        """Flag for target points that the scheme masks for unmasked source data."""
        ones = np.ma.MaskedArray(np.ones(self.source.shape, np.float32))
        return np.ma.getmaskarray(self(ones))


class GreatCircleNearestMethod(RegridMethod):
    """Take the source point nearest each target point along the Earth's surface."""

    def __init__(self, source: Grid, target: Grid) -> None:
        self._iy, self._ix, self._outside = grid_search.search_grid(source, target)

    def __call__(self, data: np.ma.MaskedArray) -> np.ma.MaskedArray:
        result = data[..., self._iy, self._ix]
        result[..., self.outside] = np.ma.masked
        return result

    @property
    def outside(self) -> NDArray[np.bool_]:
        """Flag for target points whose search starts outside the source cells."""
        return self._outside


# Builds each regridding method for a (source, target) pair
_METHODS: dict[Method, Callable[[Grid, Grid], RegridMethod]] = {
    "linear": partial(IrisMethod, Linear(extrapolation_mode="mask")),
    "nearest": partial(IrisMethod, Nearest(extrapolation_mode="mask")),
    # Area weighting uses the source cells, so is not padded
    "area_weighted": partial(IrisMethod, AreaWeighted(), pad=False),
    "nearest_great_circle": GreatCircleNearestMethod,
}
