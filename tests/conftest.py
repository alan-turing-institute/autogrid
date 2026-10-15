import pytest

from autogrid import Grid
from autogrid.grids import build_ease2, build_global_latlon


@pytest.fixture(scope="module")
def latlon() -> Grid:
    return build_global_latlon(1.0)


@pytest.fixture(scope="module")
def south() -> Grid:
    return build_ease2("south", "25p0km", shape=(432, 432))
