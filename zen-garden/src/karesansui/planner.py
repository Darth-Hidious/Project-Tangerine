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
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import nearest_points

from .config import Garden
from .geometry import (allowed_region, cumulative_length, curvature, free_space, poses_from_path,
                       poses_valid, resample, sand_region, split_runs, stone_polygons, swing_radius,
                       tine_paths)
from .patterns import (PATTERNS, PatternError, Stroke, frame_centreline, head_lateral_half, islands,
                       ring_base_offset, ring_centreline)

CURVATURE_WINDOW = 10.0   # mm, see geometry.curvature
SLACK = 0.5


@dataclass
class Pass:
    kind: str               # "screed" | "rake"
    poses: np.ndarray       # (N, 3) x, y, heading [rad], headings continuous across the program
    label: str = ""
    lift: np.ndarray | None = None   # (N,) mm above the pass's working height (the screed's exit ramp)


@dataclass
class Travel:
    poses: np.ndarray                   # (N, 3) waypoints; straight segments between them
    z: np.ndarray | None = None         # tool height above the sand per waypoint (arm machines)
    joints: np.ndarray | None = None    # joint values per waypoint (arm machines)


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
    erase_coverage: float = 0.0       # share of open grit the screed blade passes over
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
    machine: object = None

    @property
    def passes(self) -> list[Pass]:
        return [s for s in self.steps if isinstance(s, Pass)]


class PlanningError(RuntimeError):
    pass


# --------------------------------------------------------------------------- trimming
def trim(poses: np.ndarray, machine, kind: str) -> list[np.ndarray]:
    """Split a pose sequence into the longest runs the head can follow without touching anything.

    Headings come from the full path, so a trimmed piece keeps exactly the poses that were checked."""
    garden = machine.garden
    if len(poses) < 2:
        return []
    ok = machine.valid(poses, kind)
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


def _passes_from(xy: np.ndarray, closed: bool, kind: str, label: str, machine) -> list[Pass]:
    garden = machine.garden
    poses = poses_from_path(xy)
    pieces = trim(poses, machine, kind)
    if closed and len(pieces) == 1 and len(pieces[0]) == len(poses):
        looped = poses_from_path(_close_loop(xy, garden.planner.closed_overlap, garden.planner.sample_step))
        if machine.valid(looped, kind).all():
            pieces = [looped]
    return [Pass(kind, p, label) for p in pieces]


def _kept(passes: list[Pass]) -> float:
    return float(sum(cumulative_length(p.poses[:, :2])[-1] for p in passes))


def _best_open_arc(xy: np.ndarray, label: str, machine) -> list[Pass]:
    """Rake an open arc in whichever way reaches the most of it.

    The skid leads by ``reach_ahead``, so an arc that runs into a wall stops early at that
    end while the other end (where the head backs away from a wall) can start close to it.
    Candidates: forwards, backwards, or split in the middle and raked from both ends inwards,
    the second half running on past the joint far enough to re-cut what its skid flattened."""
    garden = machine.garden
    fwd = _passes_from(xy, False, "rake", label, machine)
    rev = _passes_from(xy[::-1].copy(), False, "rake", label, machine)
    best = max((fwd, rev), key=_kept)
    s = cumulative_length(xy)
    if s[-1] > 4 * garden.planner.min_stroke:
        mid = int(np.searchsorted(s, s[-1] / 2))
        over = int(np.searchsorted(s, s[mid] - garden.rake.reach_ahead - 10.0))
        split = (_passes_from(xy[: mid + 1], False, "rake", label, machine)
                 + _passes_from(xy[over:][::-1].copy(), False, "rake", label, machine))
        # A joint costs a lift mark; only take it for a real gain in raked length.
        if _kept(split) - (s[mid] - s[over]) > _kept(best) + 20.0:
            best = split
    return best


def rake_passes(machine, strokes: list[Stroke]) -> tuple[list[Pass], float]:
    passes, dropped = [], 0.0
    for st in strokes:
        label = f"{st.kind}:{st.island}" if st.island else st.kind
        if st.kind in ("ring", "spiral", "frame") and not st.closed:
            new = _best_open_arc(st.xy, label, machine)
        else:
            new = _passes_from(st.xy, st.closed, "rake", label, machine)
        dropped += max(cumulative_length(st.xy)[-1] - _kept(new), 0.0)
        passes += new
    return passes, dropped


def screed_passes(machine, bare: bool = False) -> list[Pass]:
    """Erase: a loop along the edge of an organic sand area, lanes east to west across the bed,
    then rings around every island. ``bare`` is the erase as first written (lanes and rings,
    the blade lifted at once), kept as the baseline that experiments/erase_study.py compares.

    The blade drops the sand it is still carrying where a pass ends. Lanes end at the edge, in
    the band the loop sweeps first on the next erase, and a loop runs back over its own start;
    so leftovers are carried on rather than piled up (the multi-cycle test in test_sim checks
    that the bed stays level). A blade has no inner tine to run backwards, so the loop may turn
    as tight as half the blade plus 10 mm and follow the edge more closely than the rake frame."""
    garden = machine.garden
    wc, step = garden.planner.wall_clearance, garden.planner.sample_step
    lat = head_lateral_half(garden)
    spacing = garden.screed.width - garden.screed.overlap
    passes = []
    if garden.sand_outline and not bare:
        edge = frame_centreline(garden, radius=garden.screed.width / 2 + 10.0)
        passes += _passes_from(edge, True, "screed", "screed:frame", machine)
    bx0, by0, bx1, by1 = sand_region(garden).bounds
    lo, hi = by0 + wc + lat + SLACK, by1 - wc - lat - SLACK
    ys = list(np.arange(lo, hi + 1e-9, spacing))
    if hi - ys[-1] > 1e-6:
        ys.append(hi)
    back = max(garden.rake.bar_thickness, garden.screed.thickness) / 2
    x_start = bx1 - wc - back - SLACK   # heading west: only the back of the head faces east
    x_end = bx0 + wc + garden.rake.reach_ahead + SLACK   # the skid leads towards the west wall
    for y in ys:
        xy = resample(np.array([[x_start, y], [x_end, y]]), step)
        passes += _passes_from(xy, False, "screed", "screed:lane", machine)
    for island in islands(garden):
        reach = (island.offsets[-1] if island.offsets else 0.0) + garden.rake.tine_half
        d = ring_base_offset(garden)
        while d - garden.screed.width / 2 < reach + spacing / 2:
            xy = ring_centreline(island, d, step)
            passes += _passes_from(xy, True, "screed", f"screed:ring:{island.name}", machine)
            d += spacing
    if bare:
        return passes
    loop_radius = garden.screed.width / 2 + 10.0
    passes = [_exit_turn(p, machine, loop_radius) if p.label.startswith(("screed:frame", "screed:ring")) else p
              for p in passes]
    for p in passes:
        p.lift = feather(p.poses, garden)
    return passes


def _exit_turn(p: Pass, machine, radius: float) -> Pass:
    """End a screed loop with a quarter turn to the right, as far as the head stays clear.

    Right of the edge loop (clockwise) is the open sand; right of a ring (counter-clockwise
    round its stones) is away from them. So the blade's exit ramp lays what it still carries
    down on sand a later pass sweeps, not beside the strips no pass reaches."""
    x, y, th = p.poses[-1]
    centre = np.array([x + radius * math.sin(th), y - radius * math.cos(th)])
    step = machine.garden.planner.sample_step
    k = max(int(math.ceil(radius * math.pi / 2 / step)), 2)
    phi = np.linspace(0.0, math.pi / 2, k + 1)[1:]
    v = np.array([x, y]) - centre
    pts = centre + np.column_stack([v[0] * np.cos(phi) + v[1] * np.sin(phi),
                                    -v[0] * np.sin(phi) + v[1] * np.cos(phi)])
    arc = np.column_stack([pts, th - phi])
    ok = machine.valid(arc, "screed")
    n = len(ok) if ok.all() else int(np.argmin(ok))
    if n < 2:
        return p
    return Pass(p.kind, np.vstack([p.poses, arc[:n]]), p.label)


def feather(poses: np.ndarray, garden: Garden) -> np.ndarray:
    """The blade's exit ramp: flat, then rising ``feather_rise`` over the last ``feather`` mm.

    A blade lifted at once leaves everything it carries in one pile just ahead of it, and a pile
    at the end of a pass lands in sand no pass sweeps (the edge strip, the gap between stones),
    so it grows every erase. Rising slowly, the blade lays that sand down as a thin wedge along
    its own swept band, where the next erase picks it up again."""
    s = cumulative_length(poses[:, :2])
    sc = garden.screed
    return sc.feather_rise * np.clip((s - (s[-1] - sc.feather)) / sc.feather, 0.0, 1.0)


def screed_coverage(passes: list[Pass], garden: Garden, cell: float = 2.0) -> tuple[float, np.ndarray]:
    """Share of the open sand (sand area minus stone footprints) the blade passes over."""
    W, D = garden.tray.width, garden.tray.depth
    xs, ys = np.arange(cell / 2, W, cell), np.arange(cell / 2, D, cell)
    X, Y = np.meshgrid(xs, ys)
    grit = shapely.contains_xy(sand_region(garden), X, Y)
    for poly in stone_polygons(garden):
        grit &= ~shapely.contains_xy(poly, X, Y)
    screeds = [p for p in passes if p.kind == "screed"]
    if not screeds:
        return 0.0, np.zeros(X.shape, bool)
    half = garden.screed.width / 2
    along = np.arange(-half, half + 1e-9, cell / 2)
    pts = []
    for p in screeds:
        xy = resample(p.poses[:, :2], cell / 2)
        h = poses_from_path(xy)[:, 2]
        n = np.column_stack([-np.sin(h), np.cos(h)])
        pts.append((xy[:, None, :] + along[None, :, None] * n[:, None, :]).reshape(-1, 2))
    d, _ = cKDTree(np.vstack(pts)).query(np.column_stack([X.ravel(), Y.ravel()]), distance_upper_bound=cell)
    swept = (d.reshape(X.shape) <= cell / 2 * math.sqrt(2)) & grit
    return float(swept.sum() / grit.sum()), swept


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


# --------------------------------------------------------------------------- machines
class GantryMachine:
    """A gantry over the tray: the head clears the grit when travelling but not the walls or
    stones, so travel is routed around them and the head only turns in free space."""

    name = "gantry"

    def __init__(self, garden: Garden):
        self.garden = garden
        self.region = allowed_region(garden)
        self.free = free_space(garden)
        if self.free.is_empty:
            raise PlanningError("no free space for the head to turn in")
        self.router = Router(self.free)

    def valid(self, poses: np.ndarray, kind: str) -> np.ndarray:
        return poses_valid(poses, self.garden, self.region)

    def park(self) -> np.ndarray:
        xy = np.asarray(nearest_points(self.free.buffer(-0.5), Point(0.0, self.garden.tray.depth / 2))[0].coords[0])
        return np.array([*xy, 0.0])

    def travel(self, p0, p1, kind0, kind1) -> Travel:
        return travel_between(p0, p1, self.garden, self.region, self.router)

    def check(self, steps) -> tuple[int, int, list[str]]:
        step = self.garden.planner.sample_step
        everything = np.vstack([s.poses if isinstance(s, Pass) else densify(s.poses, step) for s in steps])
        ok = poses_valid(everything, self.garden, self.region)
        bad = int((~ok).sum())
        return len(ok), bad, ([f"{bad} head poses collide with a wall or stone"] if bad else [])


class ArmMachine:
    """An arm beside the sand. The head is lifted clear of every obstacle it can reach when
    travelling, so travel is lift, a joint-space move, lower. Rake and screed poses must be
    reachable by the arm as well as clear of the sand's edge and the stones."""

    def __init__(self, garden: Garden, kind: str | None = None):
        from .arm import make_arm
        self.garden = garden
        self.name = kind or garden.arm.kind
        self.region = allowed_region(garden)
        shapely.prepare(self.region)
        self.arm = make_arm(garden, self.name)
        self.z_travel = garden.arm.travel_lift

    def z_for(self, kind: str) -> float:
        """Tool point (the head's carriage reference, as for the gantry: the blade edge) above the sand."""
        return {"rake": self.garden.gantry.rake_clearance, "screed": 0.0}.get(kind, self.z_travel)

    def ik(self, poses: np.ndarray, z, q_prev=None):
        z = np.broadcast_to(np.asarray(z, float), (len(poses),))
        return self.arm.ik(np.column_stack([poses[:, :2], z, poses[:, 2]]), q_prev)

    def valid(self, poses: np.ndarray, kind: str) -> np.ndarray:
        return poses_valid(poses, self.garden, self.region) & self.ik(poses, self.z_for(kind))[1]

    def park(self) -> np.ndarray:
        """Head lifted over the sand edge nearest the base, arm folded towards its base."""
        a = self.arm
        r = 0.5 * (swing_radius(self.garden.rake, self.garden.screed) + 0.0) + self._r_min() + 30.0
        xy = a.base + r * np.array([math.cos(a.zero), math.sin(a.zero)])
        return np.array([*xy, a.zero])

    def _r_min(self) -> float:
        a = self.arm
        if self.name == "scara":
            return math.sqrt(a.l1**2 + a.l2**2 + 2 * a.l1 * a.l2 * math.cos(a.lim2[1]))
        return 150.0

    def travel(self, p0, p1, kind0, kind1) -> Travel:
        z0, z1, zt = self.z_for(kind0), self.z_for(kind1), self.z_travel
        qa, oka = self.ik(np.array([p0]), zt)
        qb, okb = self.ik(np.array([p1]), zt, q_prev=qa[0])
        if not (oka[0] and okb[0]):
            raise PlanningError("travel end point out of the arm's reach")
        span = np.abs(qb[0] - qa[0])
        span[2 if self.name == "scara" else 1] /= 1.0
        n = max(int(np.ceil(np.degrees(span.max()) / 1.0)), 1)      # one degree per waypoint
        qs = qa[0] + np.linspace(0, 1, n + 1)[:, None] * (qb[0] - qa[0])
        mid = self.arm.fk(qs)
        xy = np.vstack([p0[:2], mid[:, :2], p1[:2]])
        th = np.concatenate([[p0[2]], mid[:, 3], [p1[2]]])
        z = np.concatenate([[z0], np.full(len(mid), zt), [z1]])
        q0, _ = self.ik(np.array([p0]), z0, q_prev=qa[0])
        q1, _ = self.ik(np.array([p1]), z1, q_prev=qb[0])
        joints = np.vstack([q0, qs, q1])
        return Travel(np.column_stack([xy, th]), z, joints)

    def check(self, steps) -> tuple[int, int, list[str]]:
        n, bad, msgs = 0, 0, []
        for s in steps:
            if isinstance(s, Pass):
                ok = self.valid(s.poses, s.kind)
                n += len(ok)
                bad += int((~ok).sum())
            else:
                ok = self.ik(s.poses, s.z)[1]
                n += len(ok)
                bad += int((~ok).sum())
        if bad:
            msgs.append(f"{bad} poses collide with the sand edge or a stone, or are out of the arm's reach")
        msgs += self.obstacle_clearance()
        return n, bad, msgs

    def obstacles(self) -> list[tuple[str, float, object]]:
        """Everything that stands up in the garden: name, top above the sand, plan-view footprint."""
        g = self.garden
        items = [(s.name, s.height - g.tray.bed_depth, Polygon(s.outline)) for s in g.stones]
        items += [(l.name, l.height, Point(l.xy).buffer(l.radius)) for l in g.lanterns]
        items += [(f.name, f.height, Polygon(f.outline)) for f in g.features
                  if f.height > 0 and f.kind in ("rock", "bridge", "tree")]
        return items

    def sweep(self, step_deg: float = 0.5):
        """Plan-view segments the arm can occupy anywhere inside its joint-limit box (cached).

        SCARA: every link1 (base-elbow) and link2 (elbow-tool) segment over a grid of q1, q2.
        Articulated: its links always lie in the vertical plane of the base angle, so in plan the
        whole arm is a ray from the base, as long as the farthest horizontal reach the shoulder
        and elbow limits allow."""
        if getattr(self, "_sweep", None) is None:
            a = self.arm
            q1 = np.arange(a.lim1[0], a.lim1[1] + 1e-9, math.radians(step_deg))
            if self.name == "scara":
                q2 = np.arange(a.lim2[0], a.lim2[1] + 1e-9, math.radians(step_deg))
                Q1, Q2 = np.meshgrid(q1, q2)
                q = np.column_stack([Q1.ravel(), Q2.ravel(), np.zeros(Q1.size), np.zeros(Q1.size)])
                tool, elbow = a.fk(q)[:, :2], a.elbows(q)[:, :2]
                base = np.broadcast_to(a.base, elbow.shape)
                self._sweep = {"head": shapely.points(tool),
                               "links": np.concatenate([
                                   shapely.linestrings(np.stack([base, elbow], axis=1)),
                                   shapely.linestrings(np.stack([elbow, tool], axis=1))]),
                               "widths": np.concatenate([
                                   np.full(len(elbow), self.garden.arm.scara.link1_width / 2),
                                   np.full(len(elbow), self.garden.arm.scara.link2_width / 2)])}
            else:
                q2 = np.arange(a.lim2[0], a.lim2[1] + 1e-9, math.radians(1.0))
                q3 = np.arange(a.lim3[0], a.lim3[1] + 1e-9, math.radians(1.0))
                Q2, Q3 = np.meshgrid(q2, q3)
                reach = float((a.lu * np.cos(Q2) + a.lf * np.cos(Q2 + Q3)).max())
                ends = a.base + reach * np.column_stack([np.cos(q1 + a.zero), np.sin(q1 + a.zero)])
                rays = shapely.linestrings(np.stack([np.broadcast_to(a.base, ends.shape), ends], axis=1))
                self._sweep = {"head": rays, "links": rays, "widths": np.full(len(rays), 25.0)}
        return self._sweep

    def sweep_clearance(self, footprint) -> tuple[float, float]:
        """Smallest plan-view gap (mm) from a footprint to where the head (its swing radius) and
        the links (their widths) can be anywhere inside the joint limits. Joint limits are what
        makes this a guarantee, so the build sets them as hard stops as well as soft limits."""
        sw = self.sweep()
        head = shapely.distance(footprint, sw["head"]).min() - swing_radius(self.garden.rake, self.garden.screed)
        links = (shapely.distance(footprint, sw["links"]) - sw["widths"]).min()
        return float(head), float(links)

    def link_height(self) -> float:
        """Lowest height the links pass at (SCARA); an articulated arm's forearm comes down to
        the tool, so any obstacle taller than the travel lift is checked against its links."""
        return self.garden.arm.scara.link_height if self.name == "scara" else self.z_travel

    def obstacle_clearance(self, margin: float = 10.0) -> list[str]:
        """Anything taller than the travel lift must stay clear of everywhere the head can go, and
        anything taller than the links' underside clear of where the links go, by ``margin``."""
        out = []
        for name, height, fp in self.obstacles():
            if height <= self.z_travel - 5.0:
                continue
            head, links = self.sweep_clearance(fp)
            if head < margin:
                out.append(f"{name} ({height:.0f} mm) is taller than the travel lift ({self.z_travel:.0f} mm) "
                           f"and only {head:.0f} mm from where the head can go")
            if height > self.link_height() - 5.0 and links < margin:
                out.append(f"{name} ({height:.0f} mm) reaches the links ({self.link_height():.0f} mm) "
                           f"and is only {links:.0f} mm from where they sweep")
        return out


def make_machine(garden: Garden, kind: str | None = None):
    if garden.arm is None:
        return GantryMachine(garden)
    return ArmMachine(garden, kind)


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
    """Share of the open grit (sand area minus stone footprints) within half a pitch of a groove.

    Grooves are sampled every ``sample_step`` along their paths, so the midpoint between two
    grooves one pitch apart can sit a hair over half a pitch from the nearest sample; half a
    sample step of tolerance keeps those midpoints raked, while a missing groove (a gap of two
    pitches, midpoint a full pitch away) still shows."""
    pts, _, _ = groove_samples(passes, garden)
    W, D = garden.tray.width, garden.tray.depth
    xs, ys = np.arange(cell / 2, W, cell), np.arange(cell / 2, D, cell)
    X, Y = np.meshgrid(xs, ys)
    grit = shapely.contains_xy(sand_region(garden), X, Y)
    for poly in stone_polygons(garden):
        grit &= ~shapely.contains_xy(poly, X, Y)
    if len(pts) == 0:
        return 0.0, np.zeros(X.shape, bool)
    d, _ = cKDTree(pts).query(np.column_stack([X.ravel(), Y.ravel()]), distance_upper_bound=garden.rake.pitch)
    raked = (d.reshape(X.shape) <= garden.rake.pitch / 2 + garden.planner.sample_step / 2) & grit
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
def build_program(garden: Garden, pattern: str = "lines", erase: bool = True, machine=None,
                  bare_erase: bool = False, **params) -> Program:
    if pattern not in PATTERNS:
        raise PatternError(f"unknown pattern {pattern!r}; choose from {sorted(PATTERNS)}")
    machine = machine or make_machine(garden)
    report = Report(pattern)

    strokes = PATTERNS[pattern](garden, **params)
    rakes, report.dropped_mm = rake_passes(machine, strokes)
    screeds = screed_passes(machine, bare_erase) if erase else []
    passes = screeds + rakes
    if not rakes:
        report.errors.append("pattern produced no rakeable strokes")

    park = machine.park()
    steps: list = []
    current, kind = park.copy(), "park"
    for p in passes:
        # Shift the whole pass by whole turns so the head rotates the short way.
        shift = 2 * math.pi * round((current[2] - p.poses[0, 2]) / (2 * math.pi))
        p.poses[:, 2] += shift
        steps.append(machine.travel(current, p.poses[0], kind, p.kind))
        steps.append(p)
        current, kind = p.poses[-1].copy(), p.kind
    park_end = np.array([*park[:2], current[2]])
    steps.append(machine.travel(current, park_end, kind, "park"))

    # ---- checks
    _curvature_checks(passes, garden, report)
    report.poses_checked, report.collisions, msgs = machine.check(steps)
    report.errors += msgs
    step = garden.planner.sample_step
    everything = np.vstack([s.poses if isinstance(s, Pass) else densify(s.poses, step) for s in steps])
    report.n_rake = sum(p.kind == "rake" for p in passes)
    report.n_screed = sum(p.kind == "screed" for p in passes)
    report.rake_mm = float(sum(cumulative_length(p.poses[:, :2])[-1] for p in passes if p.kind == "rake"))
    report.screed_mm = float(sum(cumulative_length(p.poses[:, :2])[-1] for p in passes if p.kind == "screed"))
    report.travel_mm = float(sum(cumulative_length(s.poses[:, :2])[-1] for s in steps if isinstance(s, Travel)))
    report.rotation_deg = float(np.degrees(np.abs(np.diff(everything[:, 2])).sum()))
    report.coverage, _ = coverage(rakes, garden)
    report.erase_coverage, _ = screed_coverage(screeds, garden)
    report.spacing = groove_spacing(rakes, garden)
    for kind, q in report.spacing.items():
        if q[1] < 0.8 * garden.rake.pitch:
            report.warnings.append(f"{kind}: 1% of groove spacing below {q[1]:.1f} mm "
                                   f"(pitch {garden.rake.pitch:g} mm)")
    if report.dropped_mm > 0.05 * max(report.rake_mm, 1.0):
        report.warnings.append(f"{report.dropped_mm:.0f} mm of proposed raking was unreachable and dropped")
    if garden.gantry.a_limit is not None and report.rotation_deg > garden.gantry.a_limit:
        report.errors.append("head rotation exceeds the configured cable limit")
    return Program(garden, pattern, steps, report, machine)
