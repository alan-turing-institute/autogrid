import pytest

from autogrid.grids import EASE2Grid, LatLonGrid


@pytest.fixture(scope="module")
def latlon() -> LatLonGrid:
    return LatLonGrid(1.0)


@pytest.fixture(scope="module")
def south() -> EASE2Grid:
    return EASE2Grid("south", "25p0km", shape=(432, 432))
