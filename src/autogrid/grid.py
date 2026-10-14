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

    @staticmethod
    def from_latlon(
        latitudes: ArrayLike,
        longitudes: ArrayLike,
        coord_system: CoordSystem,
        tolerance: float = 0.01,
    ) -> "Grid":
        """Initialise a Grid from the latitude and longitude of each of its points.

        The points are projected into the coordinate system, where they must lie
        on a rectilinear grid. This is useful when a grid is only known from the
        2D latitude/longitude arrays of a data file.

        Args:
            latitudes: 2D (ny, nx) array of the latitude of each grid point.
            longitudes: 2D (ny, nx) array of the longitude of each grid point.
            coord_system: The coordinate system in which the grid is rectilinear.
            tolerance: Largest allowed distance between a projected point and
                the recovered grid lines, as a fraction of the grid spacing.

        Returns:
            A plain Grid, whatever class this is called on.

        Raises:
            ValueError: If the arrays are not 2D with matching shapes, or the
                points are not on a rectilinear grid in the coordinate system.
        """
        latitudes = np.asarray(latitudes, dtype=np.float64)
        longitudes = np.asarray(longitudes, dtype=np.float64)
        if latitudes.ndim != 2 or latitudes.shape != longitudes.shape:
            msg = (
                "Latitudes and longitudes must be 2D arrays of the same shape, but "
                f"got shapes {latitudes.shape} and {longitudes.shape}."
            )
            raise ValueError(msg)
        xy = coord_system.as_cartopy_crs().transform_points(
            ccrs.Geodetic(), longitudes, latitudes
        )
        x, y = xy[..., 0], xy[..., 1]
        if isinstance(coord_system, (GeogCS, RotatedGeogCS)):
            # Keep longitudes monotonic across the dateline, e.g. 179.5, -179.5
            x = np.unwrap(x, period=360, axis=1)
        x_points, y_points = x.mean(axis=0), y.mean(axis=1)
        spacing = min(np.abs(np.diff(x_points)).min(), np.abs(np.diff(y_points)).min())
        deviation = max(
            np.abs(x - x_points[np.newaxis, :]).max(),
            np.abs(y - y_points[:, np.newaxis]).max(),
        )
        # Ensure that NaN coordinates also fail the check
        if not deviation <= tolerance * spacing:
            msg = (
                f"Points are not on a rectilinear grid in {coord_system!r}: they "
                f"deviate by up to {deviation:.3g} from the recovered grid lines, "
                f"compared with a grid spacing of {spacing:.3g}."
            )
            raise ValueError(msg)
        return Grid(x_points, y_points, coord_system)

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
