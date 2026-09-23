"""Pattern generators.

A pattern is an ordered list of rake strokes: centre-line polylines for the middle tine.
Generators only *propose* strokes; the planner trims them to what the head can reach and
refuses anything the rake cannot follow (see ``planner.py``).

Conventions used by every pattern:

* Adjacent passes sit ``rake.lane_spacing`` (= tines x pitch) apart, so the groove pitch
  continues unbroken from one pass to the next.
* Stones are ringed as *islands*: each group's convex hull is offset outwards, which keeps
  every ring parallel to the next one and gives rounded corners whose radius is the offset
  itself (always larger than the rake's hard limit by construction).
* Straight or wavy lanes are raked first and run "late" into an island's ring zone: a lane
  stops only once every tine is inside the zone, and the rings raked afterwards overwrite
  the overlap. Stopping "early" would leave unraked slivers instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import shapely
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.polygon import orient
from shapely.ops import polylabel, unary_union

from .config import Garden
from .geometry import headings, left_normals, resample, sand_region, split_runs, stone_polygons

QUAD_SEGS = 64
SLACK = 0.5   # mm kept between a planned path and the limit it was derived from (polygonisation noise)


@dataclass
class Stroke:
    xy: np.ndarray            # (N, 2) centre line in raking order
    kind: str                 # "lane" | "ring" | "spiral"
    island: str = ""
    closed: bool = False      # a full loop, raked past its start by planner.closed_overlap


@dataclass(frozen=True)
class Island:
    name: str
    hull: Polygon
    rings: int
    offsets: tuple[float, ...]    # ring centre-line offsets from the hull
    zone: Polygon                 # area covered by the ring passes, to the outer tine edge


class PatternError(ValueError):
    """The requested pattern cannot be raked with this rake."""


# --------------------------------------------------------------------------- islands
def islands(garden: Garden, rings_override: int | None = None) -> list[Island]:
    groups: dict[str, list] = {}
    for stone in garden.stones:
        groups.setdefault(stone.group or stone.name, []).append(stone)
    polys = dict(zip([s.name for s in garden.stones], stone_polygons(garden)))
    rake = garden.rake
    base = ring_base_offset(garden)
    out = []
    for name, members in groups.items():
        hull = unary_union([polys[s.name] for s in members]).convex_hull
        n = rings_override if rings_override is not None else max(s.rings for s in members)
        offsets = tuple(base + k * rake.lane_spacing for k in range(max(n, 0)))
        reach = (offsets[-1] + rake.tine_half) if offsets else garden.planner.stone_clearance
        out.append(Island(name, hull, n, offsets, hull.buffer(reach, quad_segs=QUAD_SEGS)))
    return out


def ring_base_offset(garden: Garden) -> float:
    """Offset of the innermost ring from a hull: the head's side just clears the stone."""
    return garden.planner.stone_clearance + head_lateral_half(garden) + SLACK


def ring_centreline(island: Island, offset: float, step: float) -> np.ndarray:
    """Closed counter-clockwise ring starting at its northern-most point (the side facing away
    from a viewer standing south, so the unavoidable start/stop mark is hidden)."""
    ring = orient(island.hull.buffer(offset, quad_segs=QUAD_SEGS), sign=1.0).exterior
    xy = np.asarray(ring.coords)[:-1]
    i0 = int(np.argmax(xy[:, 1]))
    xy = np.vstack([xy[i0:], xy[:i0], xy[i0:i0 + 1]])
    return resample(xy, step)


# --------------------------------------------------------------------------- helpers
def head_lateral_half(garden: Garden) -> float:
    return max(garden.rake.bar_half, garden.screed.width / 2)


def lane_margins(garden: Garden) -> tuple[float, float]:
    """Symmetric start/end margins for lanes running along x (the skid leads by reach_ahead)."""
    wc = garden.planner.wall_clearance
    back = max(garden.rake.bar_thickness, garden.screed.thickness) / 2
    m = wc + max(back, garden.rake.reach_ahead) + SLACK
    x0, _, x1, _ = sand_region(garden).bounds
    return x0 + m, x1 - m


def lane_centres(garden: Garden, inset: float = 0.0) -> list[float]:
    """Lane y positions, grooves centred in the tray, no retraced grooves.

    ``inset`` shrinks the usable band on both sides (used by waves, whose crests reach
    ``amplitude`` further out than the lane centre)."""
    wc, lat, sp = garden.planner.wall_clearance, head_lateral_half(garden), garden.rake.lane_spacing
    _, y0_, _, y1_ = sand_region(garden).bounds
    lo = y0_ + wc + lat + inset + SLACK
    hi = y1_ - wc - lat - inset - SLACK
    if hi < lo:
        raise PatternError("tray too narrow for a single lane")
    n = int(math.floor((hi - lo) / sp + 1e-9)) + 1
    y0 = (lo + hi) / 2 - (n - 1) * sp / 2
    return [y0 + k * sp for k in range(n)]


def _outer_tines(xy: np.ndarray, half: float) -> tuple[np.ndarray, np.ndarray]:
    n = left_normals(headings(xy))
    return xy + half * n, xy - half * n


def clip_late(xy: np.ndarray, zones: list[Polygon], half: float) -> list[np.ndarray]:
    """Drop the samples where *both* outer tines are inside the same (convex) zone."""
    if len(xy) < 2 or not zones:
        return [xy]
    left, right = _outer_tines(xy, half)
    drop = np.zeros(len(xy), bool)
    for zone in zones:
        shapely.prepare(zone)
        drop |= shapely.contains_xy(zone, left[:, 0], left[:, 1]) & shapely.contains_xy(zone, right[:, 0], right[:, 1])
    return [xy[a:b] for a, b in split_runs(~drop) if b - a >= 2]


def _in_own_cell(pts: np.ndarray, own: Island, others: list[Island]) -> np.ndarray:
    geoms = shapely.points(pts)
    inside = np.ones(len(pts), bool)
    d_own = shapely.distance(geoms, own.hull)
    for other in others:
        inside &= d_own <= shapely.distance(geoms, other.hull)
    return inside


def _voronoi_keep(xy: np.ndarray, own: Island, others: list[Island], half: float) -> np.ndarray:
    """Keep ring samples while any tine is still in the island's own Voronoi cell.

    Clipping "late" makes the rake bands of neighbouring islands overlap at their seam;
    the island raked last overwrites the overlap instead of both leaving a wedge unraked."""
    if not others:
        return np.ones(len(xy), bool)
    left, right = _outer_tines(xy, half)
    return (_in_own_cell(xy, own, others) | _in_own_cell(left, own, others)
            | _in_own_cell(right, own, others))


def island_rings(garden: Garden, isl: list[Island], fill: bool = False) -> list[Stroke]:
    """Ring strokes around every island; with ``fill`` the rings continue until they leave the tray."""
    step, rake = garden.planner.sample_step, garden.rake
    tray = sand_region(garden)
    strokes: list[Stroke] = []
    # Smaller islands first, the dominant island last so its rings win at shared seams.
    for island in sorted(isl, key=lambda i: i.hull.area):
        others = [o for o in isl if o is not island]
        offsets = list(island.offsets)
        if fill:
            k = len(offsets)
            base = ring_base_offset(garden)
            while True:
                d = base + k * rake.lane_spacing
                if island.hull.buffer(d - rake.tine_half).contains(tray):
                    break
                offsets.append(d)
                k += 1
        reach = tray.buffer(-(garden.planner.wall_clearance + rake.tine_half))
        shapely.prepare(reach)
        for d in offsets:
            xy = ring_centreline(island, d, step)
            keep = _voronoi_keep(xy, island, others, rake.tine_half)
            if fill:
                # Filling rings run far outside the tray; only propose what the rake band can reach.
                keep &= shapely.contains_xy(reach, xy[:, 0], xy[:, 1])
            if keep.all():
                strokes.append(Stroke(xy, "ring", island.name, closed=True))
                continue
            runs = split_runs(keep)
            # A run that wraps across the start/end of the loop is one arc. The loop's first
            # and last samples are the same point, so the continuation starts at index 1.
            if len(runs) > 1 and runs[0][0] == 0 and runs[-1][1] == len(xy):
                _, b0 = runs.pop(0)
                a1, b1 = runs.pop(-1)
                xy = np.vstack([xy, xy[1:b0]])
                runs.append((a1, b1 + b0 - 1))
            for a, b in runs:
                if b - a >= 2:
                    strokes.append(Stroke(xy[a:b], "ring", island.name))
    return strokes


# --------------------------------------------------------------------------- border frame
def frame_centreline(garden: Garden) -> np.ndarray:
    """A pass that hugs the walls: rounded rectangle, raked clockwise from the north-west.

    Rings and spirals meet the walls at shallow angles, and the leading skid stops them
    early there; a frame raked last gives the pattern a clean border instead of crescents."""
    wc, lat, rake = garden.planner.wall_clearance, head_lateral_half(garden), garden.rake
    inset = wc + lat + SLACK
    region = sand_region(garden)
    rect = region.buffer(-inset, join_style="mitre" if not garden.sand_outline else "round")
    r = rake.min_radius_soft + 20.0
    # Opening rounds convex corners to radius r, the closing after it rounds concave ones.
    rounded = rect.buffer(-r, join_style="mitre").buffer(2 * r, quad_segs=QUAD_SEGS).buffer(-r, quad_segs=QUAD_SEGS)
    if rounded.geom_type == "MultiPolygon":
        rounded = max(rounded.geoms, key=lambda g: g.area)
    rounded = orient(rounded, sign=-1.0)
    xy = np.asarray(rounded.exterior.coords)[:-1]
    i0 = int(np.argmax(xy[:, 1] - 1e-3 * xy[:, 0]))          # north edge, west end
    xy = np.vstack([xy[i0:], xy[:i0], xy[i0:i0 + 1]])
    return resample(xy, garden.planner.sample_step)


def frame_zone(garden: Garden) -> Polygon:
    """The band the frame pass rakes, from the walls to its inner tine edge."""
    inner = Polygon(frame_centreline(garden)).buffer(-garden.rake.tine_half, quad_segs=QUAD_SEGS)
    return sand_region(garden).difference(inner)


def _with_frame(strokes: list[Stroke], garden: Garden, frame: bool | None) -> list[Stroke]:
    """Add the border pass. ``None`` means: yes for an organic sand outline (a frame that
    follows its edge replaces the ragged ends straight lanes leave there), no for a tray."""
    if frame is None:
        frame = bool(garden.sand_outline)
    if not frame:
        return strokes
    zone, half = frame_zone(garden), garden.rake.tine_half
    clipped = []
    for st in strokes:
        pieces = clip_late(st.xy, [zone], half)
        closed = st.closed and len(pieces) == 1 and len(pieces[0]) == len(st.xy)
        clipped += [Stroke(p, st.kind, st.island, closed) for p in pieces]
    return clipped + [Stroke(frame_centreline(garden), "frame", closed=True)]


# --------------------------------------------------------------------------- patterns
def lines(garden: Garden, rings: int | None = None, frame: bool | None = None) -> list[Stroke]:
    """Straight lanes along the long axis, stones ringed as islands."""
    isl = islands(garden, rings)
    zones = [i.zone for i in isl if i.rings > 0]
    x0, x1 = lane_margins(garden)
    step, half = garden.planner.sample_step, garden.rake.tine_half
    strokes = []
    for y in lane_centres(garden):
        xy = resample(np.array([[x0, y], [x1, y]]), step)
        strokes += [Stroke(p, "lane") for p in clip_late(xy, zones, half)]
    return _with_frame(strokes + island_rings(garden, isl), garden, frame)


def waves(garden: Garden, wavelength: float | None = None, amplitude: float | None = None,
          phase_deg: float = 0.0, rings: int | None = None, frame: bool | None = None) -> list[Stroke]:
    """Parallel sine lanes (all in phase), stones ringed as islands.

    Each lane is the previous one shifted in y, not a true parallel curve (true parallels of
    a sine grow cusps within one lane spacing), so where the lanes slope the gap between
    neighbouring lanes shrinks to lane_spacing * cos(slope) - span. The planner reports it.
    """
    rake = garden.rake
    # Defaults scale with the rake: 3.2 lane spacings per wave, ~a tenth of one as amplitude
    # (400 mm and 12 mm for the 125 mm lanes of the tray study).
    wavelength = 3.2 * rake.lane_spacing if wavelength is None else wavelength
    amplitude = 0.096 * rake.lane_spacing if amplitude is None else amplitude
    r_min = wavelength**2 / (4 * math.pi**2 * amplitude) if amplitude > 0 else math.inf
    if r_min < rake.min_radius_hard:
        raise PatternError(
            f"wave lambda={wavelength:g} mm, A={amplitude:g} mm bends to a {r_min:.1f} mm radius; "
            f"a {rake.span + rake.tine_width:g} mm rake needs at least {rake.min_radius_hard:g} mm "
            f"(inner tines would run backwards). Use lambda >= "
            f"{2 * math.pi * math.sqrt(amplitude * rake.min_radius_soft):.0f} mm at this amplitude.")
    isl = islands(garden, rings)
    zones = [i.zone for i in isl if i.rings > 0]
    x0, x1 = lane_margins(garden)
    step, half = garden.planner.sample_step, rake.tine_half
    k, phase = 2 * math.pi / wavelength, math.radians(phase_deg)
    strokes = []
    for y in lane_centres(garden, inset=amplitude):
        x = np.linspace(x0, x1, int((x1 - x0) / 0.25) + 1)
        xy = resample(np.column_stack([x, y + amplitude * np.sin(k * x + phase)]), step)
        strokes += [Stroke(p, "lane") for p in clip_late(xy, zones, half)]
    return _with_frame(strokes + island_rings(garden, isl), garden, frame)


def ripples(garden: Garden, frame: bool = True) -> list[Stroke]:
    """Rings around every island until they fill the tray; islands meet on Voronoi seams."""
    return _with_frame(island_rings(garden, islands(garden), fill=True), garden, frame)


def open_centre(garden: Garden, frame: bool = False) -> tuple[float, float]:
    """The point of the tray farthest from walls and ring zones (pole of inaccessibility)."""
    sand = sand_region(garden)
    region = sand.difference(frame_zone(garden)) if frame else sand
    for island in islands(garden):
        region = region.difference(island.zone)
    if region.is_empty:
        # Small gardens: the frame and ring zones cover everything. Centre the spiral in the
        # sand that is free of stones instead; the rings will overwrite what they cover.
        region = sand.difference(unary_union([i.hull.buffer(garden.planner.stone_clearance) for i in islands(garden)]))
    if region.geom_type == "MultiPolygon":
        region = max(region.geoms, key=lambda g: g.area)
    c = polylabel(region, tolerance=1.0)
    return c.x, c.y


def spiral(garden: Garden, centre: tuple[float, float] | None = None,
           start_radius: float | None = None, rings: int | None = None, frame: bool = True) -> list[Stroke]:
    """Archimedean spiral (one lane spacing per turn) around an open point, islands ringed."""
    rake, step = garden.rake, garden.planner.sample_step
    cx, cy = centre if centre is not None else open_centre(garden, frame)
    r0 = start_radius if start_radius is not None else rake.min_radius_soft + 5.0
    if r0 < rake.min_radius_hard:
        raise PatternError(f"spiral start radius {r0:g} mm is below the rake's hard limit "
                           f"{rake.min_radius_hard:g} mm")
    b = rake.lane_spacing / (2 * math.pi)
    bx0, by0, bx1, by1 = sand_region(garden).bounds
    corners = np.array([[bx0, by0], [bx1, by0], [bx0, by1], [bx1, by1]])
    r_max = np.hypot(corners[:, 0] - cx, corners[:, 1] - cy).max() + rake.lane_spacing
    theta = np.linspace(0, (r_max - r0) / b, int((r_max - r0) / b * 400) + 2)
    r = r0 + b * theta
    xy = resample(np.column_stack([cx + r * np.cos(theta), cy + r * np.sin(theta)]), step)
    isl = islands(garden, rings)
    zones = [i.zone for i in isl if i.rings > 0]
    # Only propose the part of the spiral the rake band can reach inside the walls.
    reach = sand_region(garden).buffer(-(garden.planner.wall_clearance + rake.tine_half))
    inside = shapely.contains_xy(reach, xy[:, 0], xy[:, 1])
    pieces = [xy[a:b] for a, b in split_runs(inside) if b - a >= 2]
    strokes = []
    for piece in pieces:
        strokes += [Stroke(p, "spiral") for p in clip_late(piece, zones, rake.tine_half)]
    return _with_frame(strokes + island_rings(garden, isl), garden, frame)


PATTERNS = {"lines": lines, "waves": waves, "ripples": ripples, "spiral": spiral}
