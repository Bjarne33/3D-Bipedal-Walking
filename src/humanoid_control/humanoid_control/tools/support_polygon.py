"""Support polygon helpers (2D): convex hull of the grounded sole corners and point margins."""

import numpy as np


def convex_hull(points) -> np.ndarray:
    """Compute the counter-clockwise convex hull of 2D points (monotone chain).

    Args:
        points: Points with at least two columns (x, y).

    Returns:
        Hull vertices, shape (n, 2).
    """
    pts = sorted({(float(x), float(y)) for x, y in np.asarray(points, float)[:, :2]})
    if len(pts) <= 2:
        return np.array(pts)

    def cross(o, a, b):
        """Return the z component of the cross product (a - o) x (b - o).

        Args:
            o: Origin point.
            a: First point.
            b: Second point.

        Returns:
            The signed cross product.
        """
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0.0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0.0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def _segment_distance(p, a, b) -> float:
    """Compute the distance from a point to a line segment.

    Args:
        p: Point (2,).
        a: Segment start (2,).
        b: Segment end (2,).

    Returns:
        The shortest distance from `p` to the segment.
    """
    ab = b - a
    t = np.clip((p - a) @ ab / max(ab @ ab, 1e-18), 0.0, 1.0)
    return float(np.linalg.norm(p - (a + t * ab)))


def signed_margin(point, hull) -> float:
    """Compute the signed distance of `point` to the boundary of a convex polygon.

    Args:
        point: Point (x, y).
        hull: Counter-clockwise convex polygon vertices, shape (n, 2).

    Returns:
        The distance, positive inside the polygon and negative outside.
    """
    p = np.asarray(point, float)[:2]
    hull = np.asarray(hull, float)
    if len(hull) == 0:
        return -np.inf
    if len(hull) == 1:
        return -float(np.linalg.norm(p - hull[0]))
    edges = list(zip(hull, np.roll(hull, -1, axis=0)))
    dist = min(_segment_distance(p, a, b) for a, b in edges)
    if len(hull) < 3:
        return -dist
    inside = all((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1])
                 * (p[0] - a[0]) >= 0.0 for a, b in edges)
    return dist if inside else -dist


def clamp_to_hull(point, hull, margin: float = 0.0):
    """Project `point` into the convex polygon `hull`, keeping at least `margin` from edges.

    Args:
        point: 2D point.
        hull: convex hull points in counter-clockwise order.
        margin: inward safety margin [m].

    Returns:
        (projected_point, saturated) with saturated=True when clamping was needed.
    """
    p = np.asarray(point, float)[:2].copy()
    hull = np.asarray(hull, float)
    if len(hull) < 3:
        return p, False
    changed = False
    for a, b in zip(hull, np.roll(hull, -1, axis=0)):
        edge = b - a
        normal = np.array([-edge[1], edge[0]])  # inward normal for CCW polygon
        nrm = np.linalg.norm(normal)
        if nrm < 1e-18:
            continue
        normal /= nrm
        # signed inward distance to the current edge line
        d = normal @ (p - a)
        if d < margin:
            p += (margin - d) * normal
            changed = True
    return p, changed
