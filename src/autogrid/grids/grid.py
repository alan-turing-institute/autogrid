from functools import cached_property

import cartopy.crs as ccrs
import numpy as np
from numpy.typing import ArrayLike, NDArray
from pyproj import CRS, Transformer

from autogrid.crs import (
    CRSLike,
    from_lonlat_transformer,
    projection,
    supported_crs,
    to_lonlat_transformer,
)
from autogrid.grids.axes import (
    FULL_TURN,
    check_axis_points,
    guess_bounds,
    normalise_longitude,
)


class Grid:
    """A rectilinear grid of points in a given coordinate reference system (CRS).

    The grid is defined by 1D x and y points and a CRS. For geographic CRSs, x
    is longitude and y is latitude, whatever the CRS's own axis order.

    Only some CRSs are supported: latitude-longitude and rotated pole CRSs in
    degrees, and projected CRSs in metres, on the WGS 84 datum or a bare
    ellipsoid, with a Greenwich prime meridian. Reproject coordinates in any
    other CRS into one of these first.
    """

    def __init__(
        self,
        x_points: ArrayLike,
        y_points: ArrayLike,
        crs: CRSLike,
        *,
        name: str | None = None,
    ) -> None:
        """Initialise a Grid from 1D x/y points in the given CRS.

        Args:
            x_points: 1D array of x coordinates of grid points.
            y_points: 1D array of y coordinates of grid points.
            crs: The CRS of the grid, as anything pyproj.CRS.from_user_input
                accepts, e.g. "EPSG:6931", 6931 or a pyproj CRS. It must be
                one of the supported CRSs.
            name: Optional label for display, e.g. "EASE2 south 25p0km". It is
                not part of the grid's identity, so is ignored by ==.

        Raises:
            ValueError: If the CRS is not supported, or either set of points is
                not 1D, has fewer than two points, contains non-finite values or
                is not strictly monotonic.
        """
        self.crs: CRS = supported_crs(crs)
        self.name = name
        self.x_points = check_axis_points(x_points, "x")
        self.y_points = check_axis_points(y_points, "y")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Grid):
            return NotImplemented
        return (
            self.crs == other.crs
            and np.array_equal(self.x_points, other.x_points)
            and np.array_equal(self.y_points, other.y_points)
        )

    def __repr__(self) -> str:
        name = f"name={self.name!r}, " if self.name is not None else ""
        return (
            f"{type(self).__name__}({name}shape={self.shape}, "
            f"crs={self.crs.to_string()!r})"
        )

    @cached_property
    def is_circular(self) -> bool:
        """Whether the x axis wraps around the globe, as for global longitudes."""
        if not self.is_geographic:
            return False
        extent = self.x_bounds[-1, 1] - self.x_bounds[0, 0]
        return bool(np.isclose(abs(extent), FULL_TURN))

    @property
    def is_geographic(self) -> bool:
        """Whether x and y are longitude and latitude (possibly rotated)."""
        return self.crs.is_geographic

    @property
    def latitudes(self) -> NDArray[np.float64]:
        """True latitude of each grid point in degrees, with shape (ny, nx)."""
        return self.lonlat[1]

    @property
    def longitudes(self) -> NDArray[np.float64]:
        """True longitude of each grid point in degrees east, with shape (ny, nx)."""
        return self.lonlat[0]

    @cached_property
    def lonlat(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return the longitude and latitude of each grid point, with shape (ny, nx)."""
        return self.lonlat_slice(slice(None))

    @property
    def projection(self) -> ccrs.Projection:
        """The cartopy projection of this grid, e.g. for plotting map axes."""
        return projection(self.crs)

    @property
    def shape(self) -> tuple[int, int]:
        """The (ny, nx) shape of the grid."""
        return (len(self.y_points), len(self.x_points))

    @cached_property
    def trf_from_lonlat(self) -> Transformer:
        """Transformer from longitude and latitude to the grid's x and y."""
        return from_lonlat_transformer(self.crs)

    @cached_property
    def trf_to_lonlat(self) -> Transformer:
        """Transformer from the grid's x and y to longitude and latitude."""
        return to_lonlat_transformer(self.crs)

    @cached_property
    def x(self) -> NDArray[np.float64]:
        """x coordinate of each grid point, with shape (ny, nx)."""
        return np.broadcast_to(self.x_points, self.shape)

    @cached_property
    def x_bounds(self) -> NDArray[np.float64]:
        """x coordinate bounds for each grid cell, with shape (nx, 2)."""
        return guess_bounds(self.x_points)

    @cached_property
    def y(self) -> NDArray[np.float64]:
        """y coordinate of each grid point, with shape (ny, nx)."""
        return np.broadcast_to(self.y_points[:, np.newaxis], self.shape)

    @cached_property
    def y_bounds(self) -> NDArray[np.float64]:
        """y coordinate bounds for each grid cell, with shape (ny, 2)."""
        return guess_bounds(self.y_points)

    def contains(
        self, x: ArrayLike, y: ArrayLike, *, prenormalised: bool = False
    ) -> NDArray[np.bool_]:
        """Flag points that lie within the cells of the grid.

        Args:
            x: x coordinates of the points in the grid's CRS. For a geographic
                CRS, these are compared modulo a full turn of longitude.
            y: y coordinates of the points in the grid's CRS, of the same shape.
            prenormalised: Whether x has already been shifted to match the cells
                with normalise_longitude, so need not be shifted again.

        Returns:
            A boolean array with the shape of the points, which is False for
            points outside the cell bounds or with non-finite coordinates.
        """
        x = np.asarray(x, dtype=np.float64)
        if not prenormalised:
            x = normalise_longitude(x, self.x_bounds, self.crs)
        y = np.asarray(y, dtype=np.float64)
        # NaN fails every comparison, so is never inside
        inside = (y >= np.min(self.y_bounds)) & (y <= np.max(self.y_bounds))
        if self.is_circular:
            return inside & np.isfinite(x)
        return inside & (x >= np.min(self.x_bounds)) & (x <= np.max(self.x_bounds))

    def lonlat_slice(
        self, rows: slice
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Longitude and latitude of the points in a block of rows.

        Unlike latitudes and longitudes, these are not cached, so a large grid
        can be processed a block at a time.

        Args:
            rows: Slice of the rows of the grid.

        Returns:
            Tuple of (longitude, latitude) arrays in degrees, with the shape of
            the block of rows.
        """
        return self.trf_to_lonlat.transform(self.x[rows], self.y[rows])

    @classmethod
    def from_latlon(
        cls,
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

        Raises:
            ValueError: If the CRS is not supported, the arrays are not 2D with
                matching shapes, have fewer than two rows or columns, or the
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
        if min(latitudes.shape) < 2:
            msg = (
                "At least two rows and two columns of points are needed to recover "
                f"a grid, but got shape {latitudes.shape}."
            )
            raise ValueError(msg)
        crs = supported_crs(crs)
        x, y = from_lonlat_transformer(crs).transform(longitudes, latitudes)
        if crs.is_geographic:
            # Start every row on the same branch as the first, e.g. 180 and -180,
            # then keep longitudes monotonic across the dateline, e.g. 179.5, -179.5
            x[:, 0] -= FULL_TURN * np.round((x[:, 0] - x[0, 0]) / FULL_TURN)
            x = np.unwrap(x, period=FULL_TURN, axis=1)
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
        return cls(x_points, y_points, crs)
