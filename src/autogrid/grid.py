from functools import cached_property
from typing import Any

import cartopy.crs as ccrs
import numpy as np
from numpy.typing import ArrayLike, NDArray
from pyproj import CRS, Transformer

from autogrid.iris._convert import coord_system

# Anything pyproj.CRS.from_user_input accepts, e.g. "EPSG:6931", 6931 or a CRS
CRSLike = int | str | dict[str, Any] | CRS


class Grid:
    """A rectilinear grid of points in a given coordinate reference system (CRS).

    The grid is defined by 1D x and y points and a CRS. For geographic CRSs, x
    is longitude and y is latitude, whatever the CRS's own axis order.
    """

    def __init__(self, x_points: ArrayLike, y_points: ArrayLike, crs: CRSLike) -> None:
        """Initialise a Grid from 1D x/y points in the given CRS.

        Args:
            x_points: 1D array of x coordinates of grid points.
            y_points: 1D array of y coordinates of grid points.
            crs: The CRS of the grid, as anything pyproj.CRS.from_user_input
                accepts, e.g. "EPSG:6931", 6931 or a pyproj CRS.
        """
        self.crs: CRS = CRS.from_user_input(crs)
        self.x_points = np.asarray(x_points, dtype=np.float64)
        self.y_points = np.asarray(y_points, dtype=np.float64)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Grid):
            return NotImplemented
        return (
            self.crs == other.crs
            and np.array_equal(self.x_points, other.x_points)
            and np.array_equal(self.y_points, other.y_points)
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(shape={self.shape}, crs={self.crs.to_string()!r})"
        )

    @property
    def is_circular(self) -> bool:
        """Whether the x axis wraps around the globe, as for global longitudes."""
        extent = self.x_bounds[-1, 1] - self.x_bounds[0, 0]
        return self.is_geographic and bool(np.isclose(abs(extent), 360.0))

    @property
    def is_geographic(self) -> bool:
        """Whether x and y are longitude and latitude (possibly rotated)."""
        return self.crs.is_geographic

    @property
    def latitudes(self) -> NDArray[np.float64]:
        """True latitude of each grid point, with shape (ny, nx)."""
        return self._lonlat[1]

    @property
    def longitudes(self) -> NDArray[np.float64]:
        """True longitude of each grid point, with shape (ny, nx)."""
        return self._lonlat[0]

    @property
    def projection(self) -> ccrs.Projection:
        """The cartopy projection of this grid, e.g. for plotting map axes."""
        projection = coord_system(self.crs).as_cartopy_projection()
        if not isinstance(projection, ccrs.Projection):
            msg = f"CRS {self.crs.to_string()} is not a cartopy projection."
            raise ValueError(msg)
        return projection

    @property
    def shape(self) -> tuple[int, int]:
        """The (ny, nx) shape of the grid."""
        return (len(self.y_points), len(self.x_points))

    @cached_property
    def x(self) -> NDArray[np.float64]:
        """x coordinate of each grid point, with shape (ny, nx)."""
        return np.broadcast_to(self.x_points, self.shape)

    @cached_property
    def y(self) -> NDArray[np.float64]:
        """y coordinate of each grid point, with shape (ny, nx)."""
        return np.broadcast_to(self.y_points[:, np.newaxis], self.shape)

    @cached_property
    def x_bounds(self) -> NDArray[np.float64]:
        """x coordinate bounds for each grid cell, with shape (ny, nx, 2)."""
        return self.guess_bounds(self.x_points)

    @cached_property
    def y_bounds(self) -> NDArray[np.float64]:
        """y coordinate bounds for each grid cell, with shape (ny, nx, 2)."""
        return self.guess_bounds(self.y_points)

    @cached_property
    def _lonlat(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return the longitude and latitude of each grid point, with shape (ny, nx)."""
        return Transformer.from_crs(
            self.crs, self.crs.geodetic_crs, always_xy=True
        ).transform(self.x, self.y)

    @staticmethod
    def from_latlon(
        latitudes: ArrayLike,
        longitudes: ArrayLike,
        crs: CRSLike,
        tolerance: float = 0.01,
    ) -> "Grid":
        """Initialise a Grid from the latitude and longitude of each of its points.

        The points are projected into the CRS, where they must lie on a
        rectilinear grid. This is useful when a grid is only known from the
        2D latitude/longitude arrays of a data file.

        Args:
            latitudes: 2D (ny, nx) array of the latitude of each grid point.
            longitudes: 2D (ny, nx) array of the longitude of each grid point.
            crs: The CRS in which the grid is rectilinear, as for Grid.
            tolerance: Largest allowed distance between a projected point and
                the recovered grid lines, as a fraction of the grid spacing.

        Returns:
            A plain Grid, whatever class this is called on.

        Raises:
            ValueError: If the arrays are not 2D with matching shapes, or the
                points are not on a rectilinear grid in the CRS.
        """
        latitudes = np.asarray(latitudes, dtype=np.float64)
        longitudes = np.asarray(longitudes, dtype=np.float64)
        if latitudes.ndim != 2 or latitudes.shape != longitudes.shape:
            msg = (
                "Latitudes and longitudes must be 2D arrays of the same shape, but "
                f"got shapes {latitudes.shape} and {longitudes.shape}."
            )
            raise ValueError(msg)
        crs = CRS.from_user_input(crs)
        x, y = Transformer.from_crs(crs.geodetic_crs, crs, always_xy=True).transform(
            longitudes, latitudes
        )
        if crs.is_geographic:
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
                f"Points are not on a rectilinear grid in {crs.to_string()}: they "
                f"deviate by up to {deviation:.3g} from the recovered grid lines, "
                f"compared with a grid spacing of {spacing:.3g}."
            )
            raise ValueError(msg)
        return Grid(x_points, y_points, crs)

    @staticmethod
    def guess_bounds(points: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return (n, 2) cell bounds halfway between neighbouring points.

        The outer bounds are half a spacing beyond the first and last points.
        """
        if len(points) < 2:
            msg = "At least two points are needed to infer cell bounds."
            raise ValueError(msg)
        # Extrapolate one point beyond each end, so every edge is a midpoint
        first, last = 2 * points[0] - points[1], 2 * points[-1] - points[-2]
        padded = np.concatenate([[first], points, [last]])
        edges = (padded[:-1] + padded[1:]) / 2
        return np.column_stack([edges[:-1], edges[1:]])
