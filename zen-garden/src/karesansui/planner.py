"""Turn a pattern into an executable program and check it.

A program is: park -> erase (screed passes) -> rake passes -> park, with a collision-free
travel move between consecutive passes. Every pose the head ever takes is checked against
the walls and stones; the report says what was checked and what failed.

Head model (see geometry.head_outline_local): tine bar and screed blade on the rotation
axis, skid leading. While travelling the rake is latched up and the head clears the grit,
but not the walls or the stones, so travel is routed around them. The head only rotates
inside ``free_space`` (where its whole swing circle is clear); near walls and stones it
moves with a fixed heading.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

import numpy as np
import shapely
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points

from .config import Garden
from .geometry import (allowed_region, cumulative_length, curvature, free_space, poses_from_path,
                       poses_valid, resample, split_runs, stone_polygons, tine_paths, tray_box)
from .patterns import (PATTERNS, PatternError, Stroke, head_lateral_half, islands, ring_base_offset,
                       ring_centreline)

CURVATURE_WINDOW = 10.0   # mm, see geometry.curvature
SLACK = 0.5


@dataclass
class Pass:
    kind: str               # "screed" | "rake"
    poses: np.ndarray       # (N, 3) x, y, heading [rad], headings continuous across the program
    label: str = ""


@dataclass
class Travel:
    poses: np.ndarray       # (N, 3) waypoints; straight segments between them


@dataclass
class Report:
    pattern: str
    n_rake: int = 0
    n_screed: int = 0
    rake_mm: float = 0.0
    screed_mm: float = 0.0
    travel_mm: float = 0.0
    dropped_mm: float = 0.0           # proposed rake length removed because the head could not reach it
    min_radius_mm: float = math.inf   # tightest rake centre-line radius
    below_soft_mm: float = 0.0        # rake length tighter than the soft limit
    poses_checked: int = 0
    collisions: int = 0
    coverage: float = 0.0             # share of open grit within half a pitch of a groove
    spacing: dict = field(default_factory=dict)   # nearest-neighbour groove spacing percentiles
    rotation_deg: float = 0.0         # total head rotation, for cable/slip-ring sizing
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass
class Program:
    garden: Garden
    pattern: str
    steps: list            # Travel and Pass objects in execution order
    report: Report

    @property
    def passes(self) -> list[Pass]:
        return [s for s in self.steps if isinstance(s, Pass)]


class PlanningError(RuntimeError):
    pass


# --------------------------------------------------------------------------- trimming
def trim(poses: np.ndarray, garden: Garden, region) -> list[np.ndarray]:
    """Split a pose sequence into the longest runs the head can follow without touching anything.

    Headings come from the full path, so a trimmed piece keeps exactly the poses that were checked."""
    if len(poses) < 2:
        return []
    ok = poses_valid(poses, garden, region)
    pieces = []
    for a, b in split_runs(ok):
        piece = poses[a:b]
        if b - a >= 2 and cumulative_length(piece[:, :2])[-1] >= garden.planner.min_stroke:
            pieces.append(piece)
    return pieces


def _close_loop(xy: np.ndarray, overlap: float, step: float) -> np.ndarray:
    """Continue a closed loop past its start so the first grooves get re-cut and closed."""
    s = cumulative_length(xy)
    extra = xy[1: int(np.searchsorted(s, overlap)) + 1]
    return resample(np.vstack([xy, extra]), step)


def _passes_from(xy: np.ndarray, closed: bool, kind: str, label: str, garden: Garden, region) -> list[Pass]:
    poses = poses_from_path(xy)
    pieces = trim(poses, garden, region)
    if closed and len(pieces) == 1 and len(pieces[0]) == len(poses):
        looped = poses_from_path(_close_loop(xy, garden.planner.closed_overlap, garden.planner.sample_step))
        if poses_valid(looped, garden, region).all():
            pieces = [looped]
    return [Pass(kind, p, label) for p in pieces]


def _kept(passes: list[Pass]) -> float:
    return float(sum(cumulative_length(p.poses[:, :2])[-1] for p in passes))


def _best_open_arc(xy: np.ndarray, label: str, garden: Garden, region) -> list[Pass]:
    """Rake an open arc in whichever way reaches the most of it.

    The skid leads by ``reach_ahead``, so an arc that runs into a wall stops early at that
    end while the other end (where the head backs away from a wall) can start close to it.
    Candidates: forwards, backwards, or split in the middle and raked from both ends inwards,
    the second half running on past the joint far enough to re-cut what its skid flattened."""
    fwd = _passes_from(xy, False, "rake", label, garden, region)
    rev = _passes_from(xy[::-1].copy(), False, "rake", label, garden, region)
    best = max((fwd, rev), key=_kept)
    s = cumulative_length(xy)
    if s[-1] > 4 * garden.planner.min_stroke:
        mid = int(np.searchsorted(s, s[-1] / 2))
        over = int(np.searchsorted(s, s[mid] - garden.rake.reach_ahead - 10.0))
        split = (_passes_from(xy[: mid + 1], False, "rake", label, garden, region)
                 + _passes_from(xy[over:][::-1].copy(), False, "rake", label, garden, region))
        # A joint costs a lift mark; only take it for a real gain in raked length.
        if _kept(split) - (s[mid] - s[over]) > _kept(best) + 20.0:
            best = split
    return best


def rake_passes(garden: Garden, strokes: list[Stroke], region) -> tuple[list[Pass], float]:
    passes, dropped = [], 0.0
    for st in strokes:
        label = f"{st.kind}:{st.island}" if st.island else st.kind
        if st.kind in ("ring", "spiral", "frame") and not st.closed:
            new = _best_open_arc(st.xy, label, garden, region)
        else:
            new = _passes_from(st.xy, st.closed, "rake", label, garden, region)
        dropped += max(cumulative_length(st.xy)[-1] - _kept(new), 0.0)
        passes += new
    return passes, dropped


def screed_passes(garden: Garden, region) -> list[Pass]:
    """Erase: lanes east to west over the whole bed, then rings around every island."""
    wc, step = garden.planner.wall_clearance, garden.planner.sample_step
    lat = head_lateral_half(garden)
    spacing = garden.screed.width - garden.screed.overlap
    lo, hi = wc + lat + SLACK, garden.tray.depth - wc - lat - SLACK
    ys = list(np.arange(lo, hi + 1e-9, spacing))
    if hi - ys[-1] > 1e-6:
        ys.append(hi)
    back = max(garden.rake.bar_thickness, garden.screed.thickness) / 2
    x_start = garden.tray.width - wc - back - SLACK   # heading west: only the back of the head faces east
    x_end = wc + garden.rake.reach_ahead + SLACK      # the skid leads towards the west wall
    passes = []
    for y in ys:
        xy = resample(np.array([[x_start, y], [x_end, y]]), step)
        passes += _passes_from(xy, False, "screed", "screed:lane", garden, region)
    for island in islands(garden):
        reach = (island.offsets[-1] if island.offsets else 0.0) + garden.rake.tine_half
        d = ring_base_offset(garden)
        while d - garden.screed.width / 2 < reach + spacing / 2:
            xy = ring_centreline(island, d, step)
            passes += _passes_from(xy, True, "screed", f"screed:ring:{island.name}", garden, region)
            d += spacing
    return passes


# --------------------------------------------------------------------------- travel
class Router:
    """Shortest paths for the head's rotation axis inside the free space (visibility graph)."""

    def __init__(self, free):
        self.free = free
        shapely.prepare(free)
        inner = free.buffer(-1.0).simplify(0.5)
        polys = [inner] if inner.geom_type == "Polygon" else list(inner.geoms)
        nodes = []
        for poly in polys:
            for ring in [poly.exterior, *poly.interiors]:
                nodes.extend(np.asarray(ring.coords)[:-1])
        self.nodes = np.asarray(nodes)
        n = len(self.nodes)
        self.adj = [[] for _ in range(n)]
        ii, jj = np.triu_indices(n, 1)
        segs = shapely.linestrings(np.stack([self.nodes[ii], self.nodes[jj]], axis=1))
        vis = shapely.covers(free, segs)
        for i, j in zip(ii[vis], jj[vis]):
            d = float(np.hypot(*(self.nodes[i] - self.nodes[j])))
            self.adj[i].append((j, d))
            self.adj[j].append((i, d))

    def visible(self, a, b) -> bool:
        if np.allclose(a, b):
            return bool(shapely.covers(self.free, Point(a)))
        return bool(shapely.covers(self.free, LineString([a, b])))

    def route(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if self.visible(a, b):
            return np.array([a, b])
        n = len(self.nodes)
        src, dst = n, n + 1
        pts = np.vstack([self.nodes, a, b])
        extra = {src: [], dst: []}
        for k, p in ((src, a), (dst, b)):
            segs = shapely.linestrings(np.stack([np.repeat(p[None], n, 0), self.nodes], axis=1))
            for j in np.flatnonzero(shapely.covers(self.free, segs)):
                extra[k].append((int(j), float(np.hypot(*(p - self.nodes[j])))))
        # Dijkstra; edges into dst are the reverse of dst's visibility list.
        into_dst = {j: d for j, d in extra[dst]}
        dist, prev = {src: 0.0}, {}
        heap = [(0.0, src)]
        while heap:
            d, u = heapq.heappop(heap)
            if u == dst:
                break
            if d > dist.get(u, math.inf):
                continue
            nbrs = extra[src] if u == src else self.adj[u] + ([(dst, into_dst[u])] if u in into_dst else [])
            for v, w in nbrs:
                nd = d + w
                if nd < dist.get(v, math.inf):
                    dist[v], prev[v] = nd, u
                    heapq.heappush(heap, (nd, v))
        if dst not in prev:
            raise PlanningError(f"no collision-free route from {a.round(1)} to {b.round(1)}")
        path, u = [dst], dst
        while u != src:
            u = prev[u]
            path.append(u)
        return pts[path[::-1]]


def _straight_ok(p0, p1, heading, garden, region, step=2.0) -> bool:
    n = max(int(np.ceil(np.hypot(*(p1 - p0)) / step)), 1) + 1
    xy = np.linspace(p0, p1, n)
    return bool(poses_valid(np.column_stack([xy, np.full(n, heading)]), garden, region).all())


def escape(pose: np.ndarray, garden: Garden, region, router: Router) -> np.ndarray:
    """Shortest fixed-heading straight move (one or two legs) from a pose into free space."""
    p, th = pose[:2], pose[2]
    if shapely.covers(router.free, Point(p)):
        return np.array([p])
    target = np.asarray(nearest_points(router.free.buffer(-0.5), Point(p))[0].coords[0])
    if _straight_ok(p, target, th, garden, region):
        return np.array([p, target])
    # Two legs: first along or across the heading, then to the nearest free point.
    best = None
    dirs = [np.array([math.cos(a), math.sin(a)]) for a in (th, th + math.pi, th + math.pi / 2, th - math.pi / 2)]
    for d in dirs:
        for dist in np.arange(5.0, 400.0, 5.0):
            mid = p + d * dist
            if not _straight_ok(p, mid, th, garden, region):
                break
            end = np.asarray(nearest_points(router.free.buffer(-0.5), Point(mid))[0].coords[0])
            if _straight_ok(mid, end, th, garden, region):
                length = dist + np.hypot(*(end - mid))
                if best is None or length < best[0]:
                    best = (length, np.array([p, mid, end]))
                break
    if best is None:
        raise PlanningError(f"head cannot leave pose {np.round(pose, 1)} without a collision")
    return best[1]


def travel_between(p0: np.ndarray, p1: np.ndarray, garden: Garden, region, router: Router) -> Travel:
    out_leg = escape(p0, garden, region, router)
    in_leg = escape(p1, garden, region, router)[::-1]
    mid = router.route(out_leg[-1], in_leg[0])
    xy = np.vstack([out_leg, mid[1:-1], in_leg])
    keep = np.concatenate([[True], np.hypot(*np.diff(xy, axis=0).T) > 1e-9])
    xy = xy[keep]
    # Heading: fixed on the escape legs, interpolated over the free-space route.
    n_out, n_in = len(out_leg), len(in_leg)
    th = np.empty(len(xy))
    s = cumulative_length(xy)
    i0, i1 = n_out - 1, len(xy) - n_in
    th[: i0 + 1], th[i1:] = p0[2], p1[2]
    if i1 > i0:
        span = s[i1] - s[i0]
        frac = (s[i0:i1 + 1] - s[i0]) / span if span > 0 else np.linspace(0, 1, i1 - i0 + 1)
        th[i0:i1 + 1] = p0[2] + frac * (p1[2] - p0[2])
    elif not math.isclose(p0[2], p1[2], abs_tol=1e-9):
        raise PlanningError("head must rotate but has no free space to do it in")
    return Travel(np.column_stack([xy, th]))


def densify(poses: np.ndarray, step: float) -> np.ndarray:
    out = [poses[:1]]
    for a, b in zip(poses[:-1], poses[1:]):
        n = max(int(np.ceil(np.hypot(*(b[:2] - a[:2])) / step)), 1)
        t = np.linspace(0, 1, n + 1)[1:, None]
        out.append(a + t * (b - a))
    return np.vstack(out)


# --------------------------------------------------------------------------- checks
def _curvature_checks(passes: list[Pass], garden: Garden, report: Report) -> None:
    rake = garden.rake
    for p in passes:
        if p.kind != "rake" or len(p.poses) < 3:
            continue
        k = np.abs(curvature(p.poses[:, :2], CURVATURE_WINDOW))
        seg = np.gradient(cumulative_length(p.poses[:, :2]))
        report.min_radius_mm = min(report.min_radius_mm, 1 / k.max() if k.max() > 0 else math.inf)
        report.below_soft_mm += float(seg[k > 1 / rake.min_radius_soft].sum())
        if k.max() > 1 / rake.min_radius_hard:
            report.errors.append(f"{p.label}: radius {1 / k.max():.1f} mm < hard limit "
                                 f"{rake.min_radius_hard:.1f} mm (inner tine reverses)")
    if report.below_soft_mm > 0:
        report.warnings.append(f"{report.below_soft_mm:.0f} mm of raking below the soft radius "
                               f"{rake.min_radius_soft:.0f} mm (inner tine slower than half speed)")


def groove_samples(passes: list[Pass], garden: Garden, trim_ends: float = 0.0):
    """Points along every groove, with an id per groove and the kind of pass that cut it."""
    pts, ids, kinds = [], [], []
    gid = 0
    for p in passes:
        if p.kind != "rake":
            continue
        xy = p.poses[:, :2]
        s = cumulative_length(xy)
        keep = (s >= trim_ends) & (s <= s[-1] - trim_ends)
        for path in tine_paths(xy, garden.rake.tine_offsets):
            pts.append(path[keep])
            ids.append(np.full(keep.sum(), gid))
            kinds.append(np.full(keep.sum(), p.label))
            gid += 1
    if not pts:
        return np.zeros((0, 2)), np.zeros(0, int), np.zeros(0, str)
    return np.vstack(pts), np.concatenate(ids), np.concatenate(kinds)


def coverage(passes: list[Pass], garden: Garden, cell: float = 2.0) -> tuple[float, np.ndarray]:
    """Share of the open grit (tray minus stone footprints) within half a pitch of a groove."""
    pts, _, _ = groove_samples(passes, garden)
    W, D = garden.tray.width, garden.tray.depth
    xs, ys = np.arange(cell / 2, W, cell), np.arange(cell / 2, D, cell)
    X, Y = np.meshgrid(xs, ys)
    grit = np.ones(X.shape, bool)
    for poly in stone_polygons(garden):
        grit &= ~shapely.contains_xy(poly, X, Y)
    if len(pts) == 0:
        return 0.0, np.zeros(X.shape, bool)
    d, _ = cKDTree(pts).query(np.column_stack([X.ravel(), Y.ravel()]), distance_upper_bound=garden.rake.pitch)
    raked = (d.reshape(X.shape) <= garden.rake.pitch / 2) & grit
    return float(raked.sum() / grit.sum()), raked


def groove_spacing(passes: list[Pass], garden: Garden) -> dict:
    """Distance from every groove point to the nearest *other* groove cut by the same kind of pass.

    Pass ends are excluded: lanes deliberately run into the ring zones there."""
    pts, ids, kinds = groove_samples(passes, garden, trim_ends=garden.rake.reach_ahead)
    out = {}
    for kind in np.unique(kinds):
        sel = kinds == kind
        if kind.startswith("lane") or kind.startswith("spiral"):
            sel = np.char.startswith(kinds.astype(str), kind.split(":")[0])
        p, g = pts[sel], ids[sel]
        if len(np.unique(g)) < 2:
            continue
        tree = cKDTree(p)
        k = min(96, len(p))
        d, j = tree.query(p, k=k)
        other = np.where(g[j] != g[:, None], d, np.inf).min(axis=1)
        other = other[np.isfinite(other)]
        if len(other):
            out[kind] = {q: float(np.percentile(other, q)) for q in (1, 5, 50)}
    return out


# --------------------------------------------------------------------------- program
def build_program(garden: Garden, pattern: str = "lines", erase: bool = True, **params) -> Program:
    if pattern not in PATTERNS:
        raise PatternError(f"unknown pattern {pattern!r}; choose from {sorted(PATTERNS)}")
    region = allowed_region(garden)
    free = free_space(garden)
    if free.is_empty:
        raise PlanningError("no free space for the head to turn in")
    router = Router(free)
    report = Report(pattern)

    strokes = PATTERNS[pattern](garden, **params)
    rakes, report.dropped_mm = rake_passes(garden, strokes, region)
    screeds = screed_passes(garden, region) if erase else []
    passes = screeds + rakes
    if not rakes:
        report.errors.append("pattern produced no rakeable strokes")

    park_xy = np.asarray(nearest_points(free.buffer(-0.5), Point(0.0, garden.tray.depth / 2))[0].coords[0])
    steps: list = []
    current = np.array([*park_xy, 0.0])
    for p in passes:
        # Shift the whole pass by whole turns so the head rotates the short way.
        shift = 2 * math.pi * round((current[2] - p.poses[0, 2]) / (2 * math.pi))
        p.poses[:, 2] += shift
        steps.append(travel_between(current, p.poses[0], garden, region, router))
        steps.append(p)
        current = p.poses[-1].copy()
    park_end = np.array([*park_xy, current[2]])
    steps.append(travel_between(current, park_end, garden, region, router))

    # ---- checks
    _curvature_checks(passes, garden, report)
    step = garden.planner.sample_step
    everything = np.vstack([s.poses if isinstance(s, Pass) else densify(s.poses, step) for s in steps])
    ok = poses_valid(everything, garden, region)
    report.poses_checked = int(len(ok))
    report.collisions = int((~ok).sum())
    if report.collisions:
        report.errors.append(f"{report.collisions} head poses collide with a wall or stone")
    report.n_rake = sum(p.kind == "rake" for p in passes)
    report.n_screed = sum(p.kind == "screed" for p in passes)
    report.rake_mm = float(sum(cumulative_length(p.poses[:, :2])[-1] for p in passes if p.kind == "rake"))
    report.screed_mm = float(sum(cumulative_length(p.poses[:, :2])[-1] for p in passes if p.kind == "screed"))
    report.travel_mm = float(sum(cumulative_length(s.poses[:, :2])[-1] for s in steps if isinstance(s, Travel)))
    report.rotation_deg = float(np.degrees(np.abs(np.diff(everything[:, 2])).sum()))
    report.coverage, _ = coverage(rakes, garden)
    report.spacing = groove_spacing(rakes, garden)
    for kind, q in report.spacing.items():
        if q[1] < 0.8 * garden.rake.pitch:
            report.warnings.append(f"{kind}: 1% of groove spacing below {q[1]:.1f} mm "
                                   f"(pitch {garden.rake.pitch:g} mm)")
    if report.dropped_mm > 0.05 * max(report.rake_mm, 1.0):
        report.warnings.append(f"{report.dropped_mm:.0f} mm of proposed raking was unreachable and dropped")
    if garden.gantry.a_limit is not None and report.rotation_deg > garden.gantry.a_limit:
        report.errors.append("head rotation exceeds the configured cable limit")
    return Program(garden, pattern, steps, report)
