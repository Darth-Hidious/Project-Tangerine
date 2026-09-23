"""Plane geometry for rake paths: resampling, headings, curvature, tine offsets and head footprints.

A *pose* is (x, y, heading) with the heading in radians; the head moves along +heading
("forward"), the skid leads, and the tine bar lies across the direction of travel.
"""

from __future__ import annotations

import numpy as np
import shapely
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from .config import Garden, Rake, Screed


# --------------------------------------------------------------------------- polylines
def cumulative_length(xy: np.ndarray) -> np.ndarray:
    seg = np.hypot(*np.diff(xy, axis=0).T)
    return np.concatenate([[0.0], np.cumsum(seg)])


def polyline_length(xy: np.ndarray) -> float:
    return float(cumulative_length(xy)[-1])


def resample(xy: np.ndarray, step: float) -> np.ndarray:
    """Resample a polyline at (nearly) uniform arc length, keeping both end points."""
    xy = np.asarray(xy, dtype=float)
    s = cumulative_length(xy)
    keep = np.concatenate([[True], np.diff(s) > 1e-9])       # drop repeated vertices
    xy, s = xy[keep], s[keep]
    if len(xy) < 2 or s[-1] == 0.0:
        return xy.copy()
    n = max(int(np.ceil(s[-1] / step)), 1) + 1
    t = np.linspace(0.0, s[-1], n)
    return np.column_stack([np.interp(t, s, xy[:, 0]), np.interp(t, s, xy[:, 1])])


def headings(xy: np.ndarray) -> np.ndarray:
    """Unwrapped tangent heading (rad) at each vertex, central differences inside."""
    d = np.gradient(xy, axis=0)
    return np.unwrap(np.arctan2(d[:, 1], d[:, 0]))


def curvature(xy: np.ndarray, half_window: float | None = None) -> np.ndarray:
    """Signed curvature (1/mm, positive = turning left) per vertex, Menger formula on triplets.

    With ``half_window`` (mm) the triplet is (i-m, i, i+m) with m chosen so the outer points
    sit about that far away along the path. Polylines that come from polygonised arcs
    (shapely buffers) carry ~1e-4 mm vertex noise, which neighbour-only triplets at 2 mm
    spacing amplify into percent-level curvature noise; a few-mm window removes it while
    still catching kinks, because a corner of angle a still reads as roughly a / (2 * window).
    """
    xy = np.asarray(xy, dtype=float)
    n = len(xy)
    k = np.zeros(n)
    if n < 3:
        return k
    m = 1
    if half_window:
        step = cumulative_length(xy)[-1] / (n - 1)
        m = int(np.clip(round(half_window / max(step, 1e-9)), 1, (n - 1) // 2))
    a, b, c = xy[: n - 2 * m], xy[m: n - m], xy[2 * m:]
    ab, bc, ac = b - a, c - b, c - a
    cross = ab[:, 0] * bc[:, 1] - ab[:, 1] * bc[:, 0]
    denom = np.hypot(*ab.T) * np.hypot(*bc.T) * np.hypot(*ac.T)
    k[m: n - m] = np.divide(2.0 * cross, denom, out=np.zeros_like(cross), where=denom > 0)
    k[:m], k[n - m:] = k[m], k[n - m - 1]
    return k


def left_normals(theta: np.ndarray) -> np.ndarray:
    return np.column_stack([-np.sin(theta), np.cos(theta)])


def tine_paths(xy: np.ndarray, offsets) -> np.ndarray:
    """(n_tines, N, 2) paths of tines at signed lateral offsets (left positive)."""
    n = left_normals(headings(xy))
    return np.stack([xy + d * n for d in offsets])


def tine_speed_factors(xy: np.ndarray, offsets, half_window: float | None = None) -> np.ndarray:
    """(n_tines, N) tine speed relative to the rake centre; negative means the tine runs backwards."""
    k = curvature(xy, half_window)
    return np.stack([1.0 - d * k for d in offsets])


def min_radius(xy: np.ndarray, half_window: float | None = None) -> float:
    k = np.abs(curvature(xy, half_window))
    kmax = k.max() if len(k) else 0.0
    return float("inf") if kmax == 0 else float(1.0 / kmax)


# --------------------------------------------------------------------------- head footprint
def head_outline_local(rake: Rake, screed: Screed) -> np.ndarray:
    """Outline of the whole head (tine bar, skid, screed blade) in its own frame.

    u = forward, v = left. The rotation axis sits at the centre of the tine line, the
    screed blade is mounted on that axis too (it is raised above the bed while raking).
    """
    half_lat = max(rake.bar_half, screed.width / 2)
    back = max(rake.bar_thickness, screed.thickness) / 2
    skid_half = rake.skid_width / 2
    pts = np.array([
        [-back, -half_lat], [back, -half_lat],
        [rake.skid_gap, -skid_half], [rake.reach_ahead, -skid_half],
        [rake.reach_ahead, skid_half], [rake.skid_gap, skid_half],
        [back, half_lat], [-back, half_lat],
    ])
    return pts


def swing_radius(rake: Rake, screed: Screed) -> float:
    return float(np.hypot(*head_outline_local(rake, screed).T).max())


def place(local: np.ndarray, poses: np.ndarray) -> np.ndarray:
    """Transform a local outline (K, 2) to every pose (N, 3) -> (N, K, 2)."""
    x, y, th = poses[:, 0:1], poses[:, 1:2], poses[:, 2:3]
    c, s = np.cos(th), np.sin(th)
    u, v = local[:, 0][None, :], local[:, 1][None, :]
    return np.stack([x + c * u - s * v, y + s * u + c * v], axis=-1)


def head_footprints(poses: np.ndarray, garden: Garden) -> np.ndarray:
    return shapely.polygons(place(head_outline_local(garden.rake, garden.screed), poses))


# --------------------------------------------------------------------------- tray regions
def tray_box(garden: Garden) -> Polygon:
    return box(0.0, 0.0, garden.tray.width, garden.tray.depth)


def stone_polygons(garden: Garden) -> list[Polygon]:
    return [Polygon(s.outline) for s in garden.stones]


def stone_union(garden: Garden):
    polys = stone_polygons(garden)
    return unary_union(polys) if polys else Polygon()


def allowed_region(garden: Garden):
    """Where any part of the head may be: inside the walls and clear of every stone."""
    p = garden.planner
    region = tray_box(garden).buffer(-p.wall_clearance, join_style="mitre")
    stones = stone_union(garden)
    if not stones.is_empty:
        region = region.difference(stones.buffer(p.stone_clearance))
    return region


def free_space(garden: Garden):
    """Where the head's rotation axis may be while it turns freely (the whole swing circle is clear)."""
    p = garden.planner
    r = swing_radius(garden.rake, garden.screed)
    region = tray_box(garden).buffer(-(p.wall_clearance + r), join_style="mitre")
    stones = stone_union(garden)
    if not stones.is_empty:
        region = region.difference(stones.buffer(p.stone_clearance + r))
    return region


def poses_valid(poses: np.ndarray, garden: Garden, region=None) -> np.ndarray:
    """Boolean per pose: the head footprint lies entirely inside the allowed region."""
    region = allowed_region(garden) if region is None else region
    shapely.prepare(region)
    return shapely.contains(region, head_footprints(poses, garden))


def poses_from_path(xy: np.ndarray) -> np.ndarray:
    return np.column_stack([xy, headings(xy)])


def split_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True runs as (start, stop) index pairs, stop exclusive."""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    edges = np.flatnonzero(np.diff(m.astype(np.int8)))
    return list(zip(edges[0::2], edges[1::2]))
