"""Convert autogrid objects to iris objects, and iris regridding schemes."""

from functools import cache
from typing import Literal

import cartopy.crs as ccrs
import numpy as np
from iris.analysis.cartography import DEFAULT_SPHERICAL_EARTH_RADIUS
from iris.coord_systems import CoordSystem, GeogCS
from iris.coords import DimCoord
from iris.cube import Cube
from numpy.typing import ArrayLike, NDArray
from pyproj import CRS

from autogrid.adaptors.cartopy import cartopy_projection
from autogrid.grid import Grid
from autogrid.nearest import nearest_indices


class _CartopyCS(CoordSystem):
    """An iris coordinate system that delegates to cartopy.

    Iris only needs a coordinate system to provide a cartopy CRS, so this works
    for any CRS. Iris cannot save it to netCDF.
    """

    def __init__(self, crs: CRS) -> None:
        self.crs = crs
        self.grid_mapping_name = crs.to_cf().get("grid_mapping_name", "unknown")  # ty: ignore[invalid-attribute-access]

    def as_cartopy_crs(self) -> ccrs.Projection:
        return cartopy_projection(self.crs)

    def as_cartopy_projection(self) -> ccrs.Projection:
        return self.as_cartopy_crs()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.crs.to_string()!r})"


@cache
def coord_system(crs: CRS) -> CoordSystem:
    """Return the iris coordinate system for a pyproj CRS.

    Latitude-longitude CRSs must be an iris GeogCS for iris to wrap longitudes.
    Anything else is wrapped.
    """
    cf = crs.to_cf()
    if cf.get("grid_mapping_name") == "latitude_longitude":
        return GeogCS(cf["semi_major_axis"], cf["semi_minor_axis"])
    return _CartopyCS(crs)


def _dim_coord(grid: Grid, axis: Literal["x", "y"]) -> DimCoord:
    """Build an iris dimension coordinate for one axis of a Grid."""
    if grid.crs.to_cf().get("grid_mapping_name") == "rotated_latitude_longitude":
        name, units = f"grid_{'longitude' if axis == 'x' else 'latitude'}", "degrees"
    elif grid.is_geographic:
        name, units = "longitude" if axis == "x" else "latitude", "degrees"
    else:
        name, units = f"projection_{axis}_coordinate", "m"
    return DimCoord(
        grid.x_points if axis == "x" else grid.y_points,
        standard_name=name,
        units=units,
        bounds=grid.x_bounds if axis == "x" else grid.y_bounds,
        coord_system=coord_system(grid.crs),
        # Mark global longitudes as circular so regridders wrap correctly
        circular=axis == "x" and grid.is_circular,
    )


def to_cube(grid: Grid, data: ArrayLike | None = None) -> Cube:
    """Build an iris cube on a Grid, with any leading dimensions of data."""
    shape = grid.shape if data is None else np.shape(data)
    n_lead = len(shape) - 2
    return Cube(
        data,
        shape=None if data is not None else shape,
        dim_coords_and_dims=[
            (_dim_coord(grid, "y"), n_lead),
            (_dim_coord(grid, "x"), n_lead + 1),
        ],
    )


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

    def __init__(self, src_cube: Cube, target_cube: Cube) -> None:
        """Initialise the regridder.

        Args:
            src_cube: Cube whose last two dimensions are the source y and x.
            target_cube: Cube whose last two dimensions are the target y and x.
        """
        self._src_coords = self.yx_coords(src_cube)
        self._target_coords = self.yx_coords(target_cube)
        src, target = (
            self._grid(y, x) for y, x in (self._src_coords, self._target_coords)
        )
        self.indices: tuple[NDArray[np.intp], ...] = nearest_indices(
            src.latitudes, src.longitudes, target.latitudes, target.longitudes
        )

    def __call__(self, src: Cube) -> Cube:
        """Regrid a cube on the source grid onto the target grid.

        Dimension coordinates of any leading dimensions are kept; other
        coordinates are not.
        """
        if self.yx_coords(src) != self._src_coords:
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

    @staticmethod
    def _grid(y: DimCoord, x: DimCoord) -> Grid:
        """Build a Grid from y and x dimension coordinates.

        Coordinates with no coordinate system, as from netCDF files with no
        grid_mapping, are taken to be latitude and longitude on a sphere, as
        iris does. Any other coordinates must have a coordinate system.
        """
        cs = x.coord_system or y.coord_system
        if cs is not None:
            return Grid(x.points, y.points, cs.as_cartopy_crs())
        if (y.standard_name, x.standard_name) != ("latitude", "longitude"):
            msg = (
                f"The {y.standard_name!r} and {x.standard_name!r} coordinates have"
                " no coordinate system. Only latitude and longitude can be used"
                " without one."
            )
            raise ValueError(msg)
        return Grid(
            x.units.convert(x.points, "degrees"),
            y.units.convert(y.points, "degrees"),
            GeogCS(DEFAULT_SPHERICAL_EARTH_RADIUS).as_cartopy_crs(),
        )

    @staticmethod
    def yx_coords(cube: Cube) -> tuple[DimCoord, DimCoord]:
        """Return the (y, x) dimension coordinates of a cube's last two dimensions."""
        y = cube.coord(axis="Y", dim_coords=True)
        x = cube.coord(axis="X", dim_coords=True)
        if cube.coord_dims(y) != (cube.ndim - 2,) or cube.coord_dims(x) != (
            cube.ndim - 1,
        ):
            msg = "The y and x coordinates must be the last two dimensions of the cube."
            raise ValueError(msg)
        if not isinstance(y, DimCoord) or not isinstance(x, DimCoord):
            msg = "The y and x coordinates must be dimension coordinates."
            raise ValueError(msg)
        return y, x
