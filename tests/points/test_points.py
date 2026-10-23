import numpy as np
import pytest

from autogrid.points import Points


def test_points_have_shape_and_size() -> None:
    points = Points(np.zeros((3, 4)), np.ones((3, 4)), name="stations")
    assert points.shape == (3, 4)
    assert points.size == 12
    assert repr(points) == "Points(name='stations', shape=(3, 4))"


def test_points_latlon_is_read_only() -> None:
    points = Points([10.0, 20.0], [30.0, 40.0])
    lats, lons = points.latlon()
    np.testing.assert_array_equal(lats, [10.0, 20.0])
    np.testing.assert_array_equal(lons, [30.0, 40.0])
    with pytest.raises(ValueError, match="read-only"):
        lats[0] = 0.0


def test_points_equality_ignores_name() -> None:
    assert Points([1.0], [2.0], name="a") == Points([1.0], [2.0], name="b")
    assert Points([1.0], [2.0]) != Points([1.0], [3.0])


@pytest.mark.parametrize(
    ("lats", "lons", "match"),
    [
        ([1.0, 2.0], [1.0], "same shape"),
        ([], [], "must not be empty"),
        ([np.nan], [0.0], "must all be finite"),
    ],
)
def test_points_reject_invalid_latlon(
    lats: list[float], lons: list[float], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        Points(lats, lons)
