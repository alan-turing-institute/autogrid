# autogrid

Regrid geophysical fields between latitude-longitude and projected grids, such as EASE-Grid 2.0 and
between scattered points, using great circle distances where it matters.

- `Grid`: a rectilinear grid of points in any CRS, with its latitudes and longitudes.
- `Regridder`: regrid arrays from one `Grid` to another, caching the weights.
- `autogrid.regrid`: various utility methods, e.g. nearest neighbours and Gaussian weighted means of
plain latitude/longitude points, with no CRS needed.

## Example workflows

### Regrid a global field onto a polar grid

Build one `Regridder` per pair of grids and apply it to every field. Its weights are computed on
first use of each method and reused after that.

```python
import numpy as np

from autogrid import Regridder
from autogrid.grids import build_ease2, build_global_latlon

# Global 1 degree latitude-longitude grid on WGS84 (EPSG:4326)
source = build_global_latlon(1.0)

# 432 x 432 cell subset of the 25 km EASE-Grid 2.0 South grid (EPSG:6932),
# centred on the South Pole
target = build_ease2("south", "25p0km", shape=(432, 432))

# Any leading dimensions, e.g. (time, ny, nx), are regridded together
field = np.cos(np.deg2rad(source.latitudes)) * np.sin(np.deg2rad(source.longitudes))
forecast = np.stack([field, 2 * field, 3 * field])

regridder = Regridder(source, target)
regridded = regridder.apply(forecast, method="linear")  # shape (3, 432, 432)
```

The available methods are:

| Method | Description |
|---|---|
| `"linear"` | Bilinear interpolation in the source grid's own coordinates |
| `"nearest"` | Nearest source point in the source grid's own coordinates (for a latitude-longitude
source, nearest in degrees) |
| `"nearest_great_circle"` | Nearest source point along the Earth's surface |
| `"area_weighted"` | Weighted area-overlap regridding, for grids that share a CRS |

Target points outside the source grid are NaN, and so are any target points computed from NaN or
masked source values. For the `"linear"` and `"nearest"` methods, target points between the
outermost source points and the edge of the source cells take the value of the edge cell.
`regridder.outside(method)` flags the target points that fall outside the source grid for that
method.

### Regrid a forecast file from EASE-Grid 2.0 to latitude-longitude

Data files often store only 2D latitude and longitude arrays. `Grid.from_latlon` recovers the grid
by projecting those points into the CRS where the grid is rectilinear, and raises an error if they
do not lie on one.

```python
import numpy as np

from autogrid import Grid, Regridder
from autogrid.grids import build_ease2

# A synthetic sea ice dataset
ease2 = build_ease2("north", "25p0km", shape=(432, 432))
latitudes, longitudes = ease2.latitudes, ease2.longitudes
ice_conc = np.clip(np.cos(np.deg2rad(4 * (90 - latitudes))), 0, 1)[np.newaxis]

source = Grid.from_latlon(latitudes, longitudes, "EPSG:6931")
target = Grid(np.arange(-179.875, 180, 0.25), np.arange(40.125, 90, 0.25), "EPSG:4326")
regridder = Regridder(source, target)

# Regrid ignoring 'land' cells
ocean = latitudes > 60  # True for ocean, False for land
ocean_fraction = regridder.apply(ocean.astype(np.float32))
with np.errstate(divide="ignore", invalid="ignore"):
    regridded = regridder.apply(np.where(ocean, ice_conc, 0)) / ocean_fraction
regridded[..., ocean_fraction < 0.5] = np.nan  # mostly land
```

### Reproject anemoi data by nearest neighbour

`nearest_indices` needs only latitudes and longitudes, of any shape. Compute the indices once, then
apply them to any number of fields.

```python
import numpy as np

from autogrid.grids import build_ease2, build_global_latlon
from autogrid.regrid import nearest_indices

# A flat list of points, with (variable, point) data
source_grid = build_global_latlon(1.0)
source_lats = source_grid.latitudes.ravel()
source_lons = source_grid.longitudes.ravel()
data = np.random.default_rng(0).random((5, source_lats.size))

target = build_ease2("north", "25p0km", shape=(432, 432))
(index,) = nearest_indices(
    source_lats, source_lons, target.latitudes, target.longitudes
)
reprojected = data[..., index]  # shape (5, 432, 432)
```

Every target point takes its nearest source point, however far away it is, so no target points are
masked. For a 2D source, `nearest_indices` returns an `(iy, ix)` pair to use as `data[..., iy, ix]`.

### Grid scattered observations

`gaussian_mean` sets each target point to the weighted mean of nearby observations, using Gaussian
weights based on great circle distance. This suits observations whose positions change from one
entry to the next, so no weights are cached.

```python
import numpy as np

from autogrid.grids import build_ease2
from autogrid.regrid import gaussian_mean

rng = np.random.default_rng(0)
obs_lats = rng.uniform(55, 90, 500)
obs_lons = rng.uniform(-180, 180, 500)
obs_temperature = rng.normal(0, 1, 500)

target = build_ease2("north", "25p0km", shape=(432, 432))
gridded = gaussian_mean(
    obs_lats,
    obs_lons,
    obs_temperature,
    target.latitudes,
    target.longitudes,
    distance_scale_km=200,
)
```

Every observation contributes to every target point, so target points far from all observations
tend to the unweighted mean of the observations rather than to NaN.

### Find nearest points on a very large grid

A KD-tree over every point of a 1 km EASE-Grid 2.0 grid (324 million points) needs a lot of memory.
`grid_to_points` uses the grid's structure instead: it starts from the nearest point in the grid's
CRS and steps between neighbouring grid points by great circle distance. This needs memory only in
proportion to the number of target points.

```python
from autogrid.grids import build_ease2
from autogrid.regrid import grid_to_points

source = build_ease2("north", "1p0km")  # 18000 x 18000
station_lats = [78.2, 71.3, 64.8]
station_lons = [15.6, -156.8, -147.7]

iy, ix = grid_to_points(source, station_lats, station_lons)
```

The `"nearest_great_circle"` argument to `Regridder` uses this search.

### Plot a field on its own projection

`Grid.projection` is the cartopy projection of a grid's CRS, which needs the `plot` extra
(`pip install "autogrid[plot]"`). Its x and y coordinates can be plotted on map axes directly.

```python
import matplotlib.pyplot as plt
import numpy as np

from autogrid.grids import build_ease2

grid = build_ease2("south", "25p0km", shape=(432, 432))
field = np.cos(np.deg2rad(grid.latitudes)) * np.sin(np.deg2rad(grid.longitudes))

ax = plt.axes(projection=grid.projection)
ax.pcolormesh(grid.x, grid.y, field, transform=grid.projection, cmap="RdBu_r")
ax.coastlines()
plt.savefig("field.png")
```
