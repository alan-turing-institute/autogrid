from functools import cached_property

import cartopy.crs as ccrs
import numpy as np
from iris.coord_systems import CoordSystem, GeogCS, RotatedGeogCS
from iris.coords import DimCoord
from numpy.typing import ArrayLike, NDArray


class Grid:
    """A rectilinear grid of points in a given coordinate reference system (CRS).

    The grid is defined by 1D x and y points and a coordinate system.
    """

    def __init__(
        self,
        x_points: ArrayLike,
        y_points: ArrayLike,
        coord_system: CoordSystem,
    ) -> None:
        """Initialise a Grid from 1D x/y points in the given coordinate system.

        Args:
            x_points: 1D array of x coordinates of grid points.
            y_points: 1D array of y coordinates of grid points.
            coord_system: The coordinate system of the grid.

        Cell bounds are inferred from the midpoints between neighbouring
        points. Bounds are needed by conservative schemes such as
        iris.analysis.AreaWeighted.
        """
        self.coord_system: CoordSystem = coord_system
        self.x_coord: DimCoord = self._make_coord("x", x_points)
        self.y_coord: DimCoord = self._make_coord("y", y_points)

    @property
    def crs(self) -> ccrs.CRS:
        """The cartopy CRS of this grid."""
        return self.coord_system.as_cartopy_crs()

    @property
    def projection(self) -> ccrs.Projection:
        """The cartopy projection of this grid, e.g. for plotting map axes."""
        return self.coord_system.as_cartopy_projection()

    @property
    def shape(self) -> tuple[int, int]:
        """The (ny, nx) shape of the grid."""
        return (len(self.y_coord.points), len(self.x_coord.points))

    @cached_property
    def x(self) -> NDArray[np.float64]:
        """x coordinate of each grid point, with shape (ny, nx)."""
        return np.broadcast_to(self.x_coord.points, self.shape)

    @cached_property
    def y(self) -> NDArray[np.float64]:
        """y coordinate of each grid point, with shape (ny, nx)."""
        return np.broadcast_to(self.y_coord.points[:, np.newaxis], self.shape)

    @cached_property
    def _lonlat(self) -> NDArray[np.float64]:
        return ccrs.Geodetic().transform_points(self.crs, self.x, self.y)

    @property
    def longitudes(self) -> NDArray[np.float64]:
        """True longitude of each grid point, with shape (ny, nx)."""
        return self._lonlat[..., 0]

    @property
    def latitudes(self) -> NDArray[np.float64]:
        """True latitude of each grid point, with shape (ny, nx)."""
        return self._lonlat[..., 1]

    def _make_coord(self, axis: str, points: ArrayLike) -> DimCoord:
        """Build a dimension coordinate with names and units for the CRS."""
        if isinstance(self.coord_system, RotatedGeogCS):
            name = {"x": "grid_longitude", "y": "grid_latitude"}[axis]
            units = "degrees"
        elif isinstance(self.coord_system, GeogCS):
            name = {"x": "longitude", "y": "latitude"}[axis]
            units = "degrees"
        else:
            name = f"projection_{axis}_coordinate"
            units = "m"
        coord = DimCoord(
            np.asarray(points, dtype=np.float64),
            standard_name=name,
            units=units,
            coord_system=self.coord_system,
        )
        if len(coord.points) > 1:
            coord.guess_bounds()
        if axis == "x" and units == "degrees" and coord.has_bounds():
            # Mark global longitudes as circular so regridders wrap correctly
            extent = coord.bounds[-1, 1] - coord.bounds[0, 0]
            coord.circular = bool(np.isclose(extent, 360.0))
        return coord

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Grid):
            return NotImplemented
        return self.x_coord == other.x_coord and self.y_coord == other.y_coord

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(shape={self.shape}, "
            f"coord_system={self.coord_system!r})"
        )
