"""Regridding schemes that follow the iris scheme interface."""

import numpy as np
from iris.coords import DimCoord
from iris.cube import Cube
from numpy.typing import NDArray
from pyproj import Transformer
from scipy.spatial import cKDTree


def _horizontal_coords(cube: Cube) -> tuple[DimCoord, DimCoord]:
    """Return the (y, x) dimension coordinates of a cube's last two dimensions."""
    y = cube.coord(axis="y", dim_coords=True)
    x = cube.coord(axis="x", dim_coords=True)
    if cube.coord_dims(y) != (cube.ndim - 2,) or cube.coord_dims(x) != (cube.ndim - 1,):
        msg = "The y and x coordinates must be the last two dimensions of the cube."
        raise ValueError(msg)
    return y, x


def _unit_vectors(cube: Cube) -> NDArray[np.float64]:
    """Return the (n, 3) unit sphere position of each horizontal point of a cube."""
    y, x = _horizontal_coords(cube)
    xx, yy = np.meshgrid(x.points, y.points)
    # Cartopy CRSs are pyproj CRSs. Use the CRS's own datum, with no datum shift,
    # to match Grid.latitudes and Grid.longitudes
    crs = x.coord_system.as_cartopy_crs()
    lon, lat = Transformer.from_crs(crs, crs.geodetic_crs, always_xy=True).transform(
        xx, yy
    )
    lon, lat = np.deg2rad(lon), np.deg2rad(lat)
    return np.stack(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1
    ).reshape(-1, 3)


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

    The nearest source point to each target point is found once, with a KD-tree,
    and reused for every cube regridded.
    """

    def __init__(self, src_grid: Cube, target_grid: Cube) -> None:
        """Initialise the regridder.

        Args:
            src_grid: Cube whose last two dimensions are the source y and x.
            target_grid: Cube whose last two dimensions are the target y and x.
        """
        self._src_coords = _horizontal_coords(src_grid)
        self._target_coords = _horizontal_coords(target_grid)
        # Straight-line distance between points on the unit sphere increases with
        # great circle distance, so a KD-tree in 3D finds the same nearest point
        tree = cKDTree(_unit_vectors(src_grid))
        _, flat_indices = tree.query(_unit_vectors(target_grid), workers=-1)
        target_shape = tuple(len(c.points) for c in self._target_coords)
        src_shape = tuple(len(c.points) for c in self._src_coords)
        self.indices: tuple[NDArray[np.intp], NDArray[np.intp]] = np.unravel_index(
            flat_indices.reshape(target_shape), src_shape
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
