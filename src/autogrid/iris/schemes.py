"""Regridding schemes that follow the iris scheme interface."""

import numpy as np
from iris.coords import DimCoord
from iris.cube import Cube
from numpy.typing import NDArray
from pyproj import Transformer

from autogrid.nearest import nearest_indices


def _horizontal_coords(cube: Cube) -> tuple[DimCoord, DimCoord]:
    """Return the (y, x) dimension coordinates of a cube's last two dimensions."""
    y = cube.coord(axis="Y", dim_coords=True)
    x = cube.coord(axis="X", dim_coords=True)
    if cube.coord_dims(y) != (cube.ndim - 2,) or cube.coord_dims(x) != (cube.ndim - 1,):
        msg = "The y and x coordinates must be the last two dimensions of the cube."
        raise ValueError(msg)
    if not isinstance(y, DimCoord) or not isinstance(x, DimCoord):
        msg = "The y and x coordinates must be dimension coordinates."
        raise ValueError(msg)
    return y, x


def _lonlat(cube: Cube) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the longitude and latitude of each horizontal point of a cube."""
    y, x = _horizontal_coords(cube)
    xx, yy = np.meshgrid(x.points, y.points)
    # Cartopy CRSs are pyproj CRSs. Use the CRS's own datum, with no datum shift,
    # to match Grid.latitudes and Grid.longitudes
    crs = x.coord_system.as_cartopy_crs()
    return Transformer.from_crs(crs, crs.geodetic_crs, always_xy=True).transform(xx, yy)


class NearestGreatCircle:
    """Nearest neighbour regridding by great circle distance on a sphere.

    Unlike iris.analysis.Nearest, which finds the nearest point in the source
    grid's own coordinates (for a latitude-longitude source, nearest in
    degrees), this finds the nearest source point along the Earth's surface.

    Use it like an iris regridding scheme, e.g. cube.regrid(grid, scheme).
    """

    def regridder(
        self, src_grid: Cube, target_grid: Cube
    ) -> "NearestGreatCircleRegridder":
        """Build a regridder from the horizontal grid of one cube to another."""
        return NearestGreatCircleRegridder(src_grid, target_grid)

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class NearestGreatCircleRegridder:
    """Regrid cubes from one horizontal grid to another by NearestGreatCircle.

    The nearest source point to each target point is found once, with
    autogrid.nearest.nearest_indices, and reused for every cube regridded.
    """

    def __init__(self, src_grid: Cube, target_grid: Cube) -> None:
        """Initialise the regridder.

        Args:
            src_grid: Cube whose last two dimensions are the source y and x.
            target_grid: Cube whose last two dimensions are the target y and x.
        """
        self._src_coords = _horizontal_coords(src_grid)
        self._target_coords = _horizontal_coords(target_grid)
        src_lons, src_lats = _lonlat(src_grid)
        target_lons, target_lats = _lonlat(target_grid)
        self.indices: tuple[NDArray[np.intp], ...] = nearest_indices(
            src_lats, src_lons, target_lats, target_lons
        )

    def __call__(self, src: Cube) -> Cube:
        """Regrid a cube on the source grid onto the target grid.

        Dimension coordinates of any leading dimensions are kept; other
        coordinates are not.
        """
        if _horizontal_coords(src) != self._src_coords:
            msg = "The cube is not on the source grid of this regridder."
            raise ValueError(msg)
        iy, ix = self.indices
        n_lead = src.ndim - 2
        result = Cube(
            src.data[..., iy, ix],
            dim_coords_and_dims=[
                *(
                    (coord.copy(), src.coord_dims(coord)[0])
                    for coord in src.dim_coords
                    if src.coord_dims(coord)[0] < n_lead
                ),
                (self._target_coords[0].copy(), n_lead),
                (self._target_coords[1].copy(), n_lead + 1),
            ],
        )
        result.metadata = src.metadata
        return result
