# autogrid

Regrid geophysical fields between latitude-longitude and projected grids, such as EASE-Grid 2.0 and
between scattered points, using great circle distances where it matters.

- `Grid`: a rectilinear grid of points in any CRS, with its latitudes and longitudes.
- `Regridder`: regrid arrays from one `Grid` to another with one method, reusing the weights.
- `grid_to_points`: sample a `Grid` at a set of points
- `points_to_grid`: project a set of scattered points onto a `Grid`

## Which function?

| From \ To | `Grid` | Points |
|---|---|---|
| **`Grid`** | `Regridder` | `grid_to_points` |
| **Points** | `points_to_grid` | `autogrid.points.nearest_indices` or `autogrid.points.gaussian_weighted_mean` |

Points are latitudes and longitudes of any shape, such as observation sites or an anemoi dataset's
flat list of points.

## Example workflows

### Regrid a forecast file from EASE-Grid 2.0 to latitude-longitude

Build one `Regridder` per pair of grids and method, and apply it to every field. Its weights are
computed when it is created and reused after that.

Data files often store only 2D latitude and longitude arrays. `Grid.from_latlon` recovers the grid
by projecting those points into the CRS where the grid is rectilinear, and raises an error if they
do not lie on one. Grids can also be built directly, e.g. with `build_ease2` or
`build_global_latlon` from `autogrid.grids`.

```python
import numpy as np

from autogrid import Grid, Regridder
from autogrid.grids import build_ease2

# A synthetic sea ice forecast on a subset of the 25 km EASE-Grid 2.0 North grid, as
# a data file would store it, with 2D latitudes and longitudes and (time, ny, nx) data
latitudes, longitudes = build_ease2("north", "25p0km", shape=(432, 432)).latlon()
ice_conc = np.clip(np.cos(np.deg2rad(4 * (90 - latitudes))), 0, 1)
forecast = np.stack([ice_conc, 0.9 * ice_conc, 0.8 * ice_conc])

source = Grid.from_latlon(latitudes, longitudes, "EPSG:6931")
target = Grid(np.arange(-179.875, 180, 0.25), np.arange(40.125, 90, 0.25), "EPSG:4326")
regridder = Regridder(source, target, method="linear")

# Any leading dimensions are regridded together
regridded = regridder.apply(forecast)  # shape (3, 200, 1440)

# Regrid ignoring 'land' cells
ocean = latitudes > 60  # True for ocean, False for land
ocean_fraction = regridder.apply(ocean.astype(np.float32))
with np.errstate(divide="ignore", invalid="ignore"):
    regridded = regridder.apply(np.where(ocean, forecast, 0)) / ocean_fraction
regridded[..., ocean_fraction < 0.5] = np.nan  # mostly land
```

The available methods are:

| Method | Description |
|---|---|
| `"linear"` | Bilinear interpolation in the source grid's own coordinates |
| `"nearest"` | Nearest source point in the source grid's own coordinates (for a latitude-longitude
source, nearest in degrees) |
| `"nearest_great_circle"` | Nearest source point along the Earth's surface |
| `"area_weighted"` | Weighted area-overlap regridding, for grids that share a CRS |

`method` can also be a function `(source, target) -> scipy.sparse.csr_array` that builds your own
weights, with shape `(target.size, source.size)`. See `autogrid.regrid.methods` for the rules the
weights must follow.

Target points outside the source grid are NaN, and so are any target points computed from NaN
source values. Only NaN marks missing values, so fill a masked array with `.filled(np.nan)` first.
For the `"linear"` and `"nearest"` methods, target points between the outermost source points and
the edge of the source cells take the value of the edge cell.

### Sample a gridded forecast at observation sites

`grid_to_points` takes the value of the nearest grid point, along the Earth's surface, to each
site. It wraps `autogrid.points.nearest_indices`.

```python
import numpy as np

from autogrid import grid_to_points
from autogrid.grids import build_ease2

grid = build_ease2("north", "25p0km", shape=(432, 432))
latitudes, longitudes = grid.latlon()
forecast = np.stack([np.cos(np.deg2rad(4 * (90 - latitudes)))] * 3)  # (time, ny, nx)

rng = np.random.default_rng(0)
station_lats = rng.uniform(60, 90, 50)
station_lons = rng.uniform(-180, 180, 50)
# Shape (3, 50)
at_stations = grid_to_points(
    grid, forecast, target_lats=station_lats, target_lons=station_lons
)
```

Every site takes its nearest grid point, however far away it is, so sites outside the grid are
not masked.

### Grid scattered observations

`points_to_grid` sets each grid point to the weighted mean of nearby observations, using Gaussian
weights based on great circle distance. This suits observations whose positions change from one
entry to the next, so no weights are cached. It wraps `autogrid.points.gaussian_weighted_mean`,
which takes target latitudes and longitudes of any shape instead of a `Grid`.

```python
import numpy as np

from autogrid import points_to_grid
from autogrid.grids import build_ease2

rng = np.random.default_rng(0)
obs_lats = rng.uniform(55, 90, 500)
obs_lons = rng.uniform(-180, 180, 500)
obs_temperature = rng.normal(0, 1, 500)

target = build_ease2("north", "25p0km", shape=(432, 432))
gridded = points_to_grid(
    target,
    source_lats=obs_lats,
    source_lons=obs_lons,
    data=obs_temperature,
    distance_scale_km=200,
)
```

Every observation contributes to every target point, so target points far from all observations
take the values of the nearest ones, however far away they are. Only NaN marks missing
observations, so fill a masked array with `.filled(np.nan)` first.

### Convert data between sets of points

`nearest_indices` needs only latitudes and longitudes, of any shape. Compute the indices once, then
apply them to any number of fields.

```python
import numpy as np

from autogrid.points import nearest_indices

rng = np.random.default_rng(0)

# Data stored as a flat list of 500 points spread evenly over the globe, with (variable, point) data
source_lats = np.rad2deg(np.arcsin(rng.uniform(-1, 1, 500)))
source_lons = rng.uniform(-180, 180, 500)
data = rng.random((5, 500))

# The points of a coarser dataset
target_lats = np.rad2deg(np.arcsin(rng.uniform(-1, 1, 250)))
target_lons = rng.uniform(-180, 180, 250)
(index,) = nearest_indices(
    source_lats=source_lats,
    source_lons=source_lons,
    target_lats=target_lats,
    target_lons=target_lons,
)
converted = data[..., index]  # shape (5, 250)
```

Every target point takes its nearest source point, however far away it is, so no target points are
masked. For a 2D source, `nearest_indices` returns an `(iy, ix)` pair to use as `data[..., iy, ix]`.
The targets can also be the points of a `Grid`, from `grid.latlon()`, to take the nearest source
point at each grid point. `gaussian_weighted_mean` takes the same latitudes and longitudes for a
smoother result.

### Plot a field on its own projection

`autogrid.plot.projection` gives the cartopy projection of a grid's CRS, which needs the `plot`
extra (`pip install "autogrid[plot]"`). Its x and y points can be plotted on map axes directly.

```python
import matplotlib.pyplot as plt
import numpy as np

from autogrid.grids import build_ease2
from autogrid.plot import projection

grid = build_ease2("south", "25p0km", shape=(432, 432))
latitudes, longitudes = grid.latlon()
field = np.cos(np.deg2rad(latitudes)) * np.sin(np.deg2rad(longitudes))
grid_projection = projection(grid)

ax = plt.axes(projection=grid_projection)
ax.pcolormesh(
    grid.x_points, grid.y_points, field, transform=grid_projection, cmap="RdBu_r"
)
ax.coastlines()
plt.savefig("field.png")
```
