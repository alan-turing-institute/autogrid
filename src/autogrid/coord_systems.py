"""Coordinate systems shared between grids."""

import cartopy.crs as ccrs
from iris.coord_systems import CoordSystem, GeogCS

WGS84 = GeogCS(6378137.0, inverse_flattening=298.257223563)


class EASE2GlobalCS(CoordSystem):
    """EPSG:6933: WGS84 Lambert cylindrical equal area, true scale at 30N/S.

    Iris has no cylindrical equal area coordinate system, so this delegates to
    cartopy's EPSG definition. Iris cannot save it to netCDF.
    """

    grid_mapping_name = "lambert_cylindrical_equal_area"

    def __init__(self) -> None:
        self.standard_parallel = 30.0
        self.longitude_of_central_meridian = 0.0
        self.false_easting = 0.0
        self.false_northing = 0.0
        self.ellipsoid = WGS84

    def as_cartopy_crs(self) -> ccrs.Projection:
        return ccrs.epsg(6933)

    def as_cartopy_projection(self) -> ccrs.Projection:
        return ccrs.epsg(6933)

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"
