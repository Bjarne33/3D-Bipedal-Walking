import numpy as np
import pytest

from humanoid_control.gait.support_polygon import clamp_to_hull, convex_hull, signed_margin

SQUARE = [[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5]]


def test_hull_drops_interior_points_and_is_ccw():
    hull = convex_hull(SQUARE)
    assert len(hull) == 4
    area = 0.5 * sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(hull, np.roll(hull, -1, axis=0)))
    assert area == pytest.approx(1.0)  # positive: counter-clockwise


def test_margin_inside_outside_and_on_edge():
    hull = convex_hull(SQUARE)
    assert signed_margin([0.5, 0.5], hull) == pytest.approx(0.5)
    assert signed_margin([0.9, 0.5], hull) == pytest.approx(0.1)
    assert signed_margin([1.2, 0.5], hull) == pytest.approx(-0.2)
    assert signed_margin([2.0, 2.0], hull) == pytest.approx(-np.sqrt(2.0))
    assert signed_margin([1.0, 0.3], hull) == pytest.approx(0.0)


def test_two_feet_hull_covers_the_gap_between_them():
    left = np.array([[-0.1, 0.05], [0.1, 0.05], [0.1, 0.25], [-0.1, 0.25]])
    right = left * [1, -1]
    hull = convex_hull(np.vstack([left, right]))
    assert signed_margin([0.0, 0.0], hull) == pytest.approx(0.1)
    assert signed_margin([0.0, 0.0], convex_hull(left)) == pytest.approx(-0.05)


def test_degenerate_hulls_count_as_outside():
    assert signed_margin([0, 0], convex_hull([[1, 0]])) == pytest.approx(-1.0)
    assert signed_margin([0.5, 1], convex_hull([[0, 0], [1, 0]])) == pytest.approx(-1.0)


def test_clamp_to_hull_keeps_inside_point():
    hull = convex_hull(SQUARE)
    p, saturated = clamp_to_hull([0.7, 0.7], hull, margin=0.1)
    assert not saturated
    np.testing.assert_allclose(p, [0.7, 0.7])


def test_clamp_to_hull_projects_outside_point():
    hull = convex_hull(SQUARE)
    p, saturated = clamp_to_hull([1.2, 0.5], hull)
    assert saturated
    np.testing.assert_allclose(p, [1.0, 0.5])


def test_clamp_to_hull_applies_margin():
    hull = convex_hull(SQUARE)
    p, saturated = clamp_to_hull([0.98, 0.5], hull, margin=0.05)
    assert saturated
    np.testing.assert_allclose(p, [0.95, 0.5], atol=1e-9)
