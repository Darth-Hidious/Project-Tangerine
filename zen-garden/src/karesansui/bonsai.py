"""The bonsai for the pictures: an informal-upright pine grown to fit the model's tree.

The model knows the tree only as a height and a canopy outline, which is what the arm must keep
clear of. The 3D viewer and the photoreal renderer both draw the tree grown here:

  - a trunk that tapers strongly, flares into surface roots, and moves in S-curves that die
    away towards the apex;
  - primary branches spiralling up at the golden angle (none pointing at the viewer in the lower
    third, as a bonsai is styled), the lowest the longest, each drooping a little and levelling
    out, ramifying into alternating side twigs that fork again;
  - pine shoots where a pine carries its needles: a rosette at the end of every twig and a few
    along its outer half, pointing out and up, so each branch carries a lumpy, domed pad that is
    ragged and shaded underneath, with the twigs showing.

Every needle tip of every shoot, placed exactly as the pictures place it (shoot_frames), stays
inside the canopy outline and under the tree's height, so the picture never shows more tree than
the clearance check covers. Lengths in mm; heights above the sand.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import shapely
from shapely.geometry import Polygon

from .config import Garden

GOLDEN = math.radians(137.5)
NEEDLE = 13.0                 # mm: needle length in a tuft (a pine's needles, reduced on a bonsai)


@dataclass
class Bonsai:
    base: tuple[float, float, float]                        # trunk base: x, y, height above the sand
    branches: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)   # (points (n, 3), radii (n,))
    tufts: list[list[float]] = field(default_factory=list)  # x, y, z, axis x, y, z, needle length, twist

    @property
    def tuft_array(self) -> np.ndarray:
        return np.asarray(self.tufts, float).reshape(-1, 8)

    def needle_tips(self) -> np.ndarray:
        """The tip of every needle of every shoot, as drawn: (tufts x needles, 3)."""
        return _tips(self.tuft_array)


def shoot_frames(tufts: np.ndarray) -> np.ndarray:
    """Rotations (n, 3, 3) that stand the unit shoot of tuft_mesh (stem along +z) on each tuft's
    axis, turned by its twist about it. The renderers place the shoots with these."""
    t = np.asarray(tufts, float).reshape(-1, 8)
    a = t[:, 3:6] / np.linalg.norm(t[:, 3:6], axis=1, keepdims=True)
    ref = np.where(np.abs(a[:, 2:3]) < 0.95, [[0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0]])
    u = np.cross(ref, a)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    v = np.cross(a, u)
    c, s = np.cos(t[:, 7:8]), np.sin(t[:, 7:8])
    return np.stack([c * u + s * v, c * v - s * u, a], axis=2)          # columns: where x, y, z go


def _tips(tufts: np.ndarray) -> np.ndarray:
    t = np.asarray(tufts, float).reshape(-1, 8)
    local = np.einsum("nij,mj->nmi", shoot_frames(t), SHOOT_TIPS) * t[:, 6, None, None]
    return (t[:, None, :3] + local).reshape(-1, 3)


def _spline(ctrl, n):
    """Catmull-Rom through the control points, n samples."""
    P = np.asarray(ctrl, float)
    P = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    segs = len(P) - 3
    out = []
    for s in range(segs):
        p0, p1, p2, p3 = P[s:s + 4]
        k = max(2, n // segs)
        for t in np.linspace(0, 1, k, endpoint=(s == segs - 1)):
            out.append(0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    return np.array(out)


def _along(pts, s):
    """Point and unit direction at arc-length fraction s of a polyline."""
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    target = s * cum[-1]
    i = min(int(np.searchsorted(cum, target, side="right")) - 1, len(seg) - 1)
    f = (target - cum[i]) / max(seg[i], 1e-9)
    d = pts[i + 1] - pts[i]
    return pts[i] + f * d, d / max(np.linalg.norm(d), 1e-9)


def _turn(v, ang):
    c, s = math.cos(ang), math.sin(ang)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def grow(garden: Garden, ground_at, seed: int = 7) -> Bonsai:
    """Grow the tree on `ground_at(x, y)` (mm above the sand) to fit the garden's tree feature."""
    tree = next(f for f in garden.features if f.kind == "tree")
    pot = next(f for f in garden.features if f.kind == "pot")
    canopy = Polygon(tree.outline)
    cx, cy = canopy.centroid.x, canopy.centroid.y
    x0, y0, x1, y1 = canopy.bounds
    ax, ay = (x1 - x0) / 2, (y1 - y0) / 2
    bx, by = Polygon(pot.outline).centroid.coords[0]
    H = tree.height
    z0 = ground_at(bx, by) - 3.0
    rng = np.random.default_rng(seed)
    tree_out = Bonsai(base=(bx, by, z0))

    # ---- trunk: taper, root flare, S-curves that die away towards the apex
    t = np.linspace(0, 1, 72)
    ease = np.clip(t / 0.15, 0, 1) ** 2 * (3 - 2 * np.clip(t / 0.15, 0, 1))
    amp = 0.09 * H * (1 - t) ** 0.6 * ease
    wob = np.column_stack([amp * np.sin(2 * np.pi * 1.1 * t + 0.6), 0.45 * amp * np.sin(2 * np.pi * 0.8 * t + 2.0)])
    plan = np.array([bx, by]) + np.outer(t ** 1.5, [cx - bx, cy - by]) + wob + np.outer(t ** 2, [0, -0.04 * H])
    trunk = np.column_stack([plan, z0 + 0.86 * H * t])
    rb = H / 12
    r_trunk = rb * (1 - 0.85 * t) ** 1.3 * (1 + 0.5 * np.exp(-t / 0.035))
    # where the ground falls away under the flare (the tree stands on a slope), the base carries on
    # down into it, widening, instead of standing on the slope with its underside open
    ring = [ground_at(bx + 1.25 * r_trunk[0] * math.cos(a), by + 1.25 * r_trunk[0] * math.sin(a))
            for a in np.linspace(0, 2 * np.pi, 24, endpoint=False)]
    z_low = min(min(ring), z0) - 3.0
    below = np.linspace(z_low, z0, 5)[:-1]
    foot = np.column_stack([np.full((4, 2), [bx, by]), below])
    r_foot = r_trunk[0] * (1 + 0.25 * (z0 - below) / max(z0 - z_low, 1e-9))
    tree_out.branches.append((np.vstack([foot, trunk]), np.concatenate([r_foot, r_trunk])))

    def trunk_at(tt):
        i = min(int(tt * (len(t) - 1)), len(t) - 2)
        f = tt * (len(t) - 1) - i
        return trunk[i] + f * (trunk[i + 1] - trunk[i]), r_trunk[i] + f * (r_trunk[i + 1] - r_trunk[i])

    # ---- surface roots (nebari): out of the flare, tapering fast, sunk in the moss to their crowns
    for k in range(9):
        th = 2 * np.pi * k / 9 + rng.uniform(-0.25, 0.25)
        s = np.linspace(0, 1, 12)
        dist = r_trunk[0] * 0.5 + rb * rng.uniform(1.6, 2.4) * s          # from the trunk's axis
        px, py = bx + np.cos(th) * dist, by + np.sin(th) * dist
        rr = rb * (0.36 * (1 - s) ** 1.4 + 0.06)
        pz = np.array([ground_at(x, y) for x, y in zip(px, py)]) + 0.2 * rr
        tree_out.branches.append((np.column_stack([px, py, pz]), rr))

    def reach(p, d):
        """How far from p along d before the canopy ellipse."""
        qx, qy = (p[0] - cx) / ax, (p[1] - cy) / ay
        dx, dy = d[0] / ax, d[1] / ay
        a, b, c = dx * dx + dy * dy, 2 * (qx * dx + qy * dy), qx * qx + qy * qy - 1
        return max((-b + math.sqrt(max(b * b - 4 * a * c, 0))) / (2 * a), 0.0)

    def twig(q, d2, length, r0, parts, depth, rise):
        """A side shoot: level or rising a little, forking again while it can."""
        side = np.array([-d2[1], d2[0]])
        wig = rng.uniform(-0.12, 0.12)
        ctrl = [q, q + np.r_[d2 * 0.5 * length + side * wig * length, rise * 0.4 * length],
                q + np.r_[d2 * length + side * wig * 0.5 * length, rise * length]]
        pts = _spline(ctrl, 7)
        parts.append((pts, np.linspace(r0, max(0.25, r0 * 0.35), len(pts))))
        if depth < 2:
            for j, s in enumerate((0.5, 0.82)):
                p2, u2 = _along(pts, s)
                h = u2[:2] / max(np.linalg.norm(u2[:2]), 1e-9)
                sgn = 1 if (j + depth + int(wig > 0)) % 2 == 0 else -1
                twig(p2, _turn(h, sgn * rng.uniform(0.6, 0.95)), (0.5 if depth == 0 else 0.55) * length,
                     max(0.25, r0 * 0.55), parts, depth + 1, rise + rng.uniform(0.0, 0.12))

    def foliage(parts, n_bare, size=NEEDLE):
        """Shoots where a pine carries its needles: a rosette at the end of every twig (a leading
        shoot and a ring round it), a crown of older shoots grown up from the twig ends towards the
        middle of the pad, and a few shoots along the outer half of the twigs (not of the first
        n_bare parts, the branch itself). They point out and up, flatter at the rim of the pad and
        steeper in its middle: a lumpy dome on top, ragged and shaded underneath. The crown twigs
        are added to `parts`."""
        ends = np.array([tw[-1] for tw, _ in parts])
        c = ends[:, :2].mean(0)
        R = np.linalg.norm(ends[:, :2] - c, axis=1).max() + 1e-9
        tufts = []

        def shoot(q, d, elev, length):
            axis = np.r_[d * math.cos(elev), math.sin(elev)] + rng.normal(0, 0.1, 3)
            tufts.append([*q, *(axis / np.linalg.norm(axis)), length, rng.uniform(0, 2 * np.pi)])

        def rosette(q, h, dome, lift):
            shoot(q, h, math.radians(30 + 35 * dome + lift), size * rng.uniform(0.95, 1.1))
            n = int(rng.integers(4, 7))
            for j in range(n):
                shoot(q, _turn(h, 2 * np.pi * j / n + rng.uniform(-0.3, 0.3)),
                      math.radians(rng.uniform(12, 40) + 30 * dome + lift), size * rng.uniform(0.7, 0.92))

        crowns = []
        for i, (tw, _) in enumerate(parts):
            tip, u = _along(tw, 1.0)
            h = u[:2] / max(np.linalg.norm(u[:2]), 1e-9)
            off = (tip[:2] - c) / R
            dome = float(np.clip(1 - off @ off, 0, 1))            # 1 in the middle of the pad, 0 at its rim
            rosette(tip, h, dome, 0.0)
            if dome > 0.25:
                # a shoot of earlier years grown up out of the twig end, with its own rosette: the
                # pad's crown, highest in its middle
                up = size * (0.5 + 0.7 * dome) * rng.uniform(0.8, 1.2)
                top = tip + np.r_[h * 0.35 * up, up]
                crowns.append((np.array([tip, 0.5 * (tip + top), top]), np.array([0.5, 0.4, 0.3])))
                rosette(top, h, dome, 15.0)
            if i < n_bare:
                continue
            seg = np.linalg.norm(np.diff(tw, axis=0), axis=1).sum()
            for k, s in enumerate(np.linspace(0.5, 0.88, max(1, int(seg / 9)))):
                q, u2 = _along(tw, s)
                h2 = u2[:2] / max(np.linalg.norm(u2[:2]), 1e-9)
                shoot(q, _turn(h2, (1 if k % 2 else -1) * rng.uniform(0.5, 1.2)),
                      math.radians(rng.uniform(35, 70)), size * rng.uniform(0.75, 0.95))
        parts += crowns
        return tufts

    def branch(p, d, L, r_start):
        """A primary branch with its ramified side shoots and its pad; returns (parts, tufts)."""
        parts = []
        side = np.array([-d[1], d[0]])
        start = p + np.r_[d * r_start * 0.5, 0]
        ctrl = [start, start + np.r_[d * 0.33 * L + side * rng.uniform(-0.06, 0.06) * L, -0.07 * L],
                start + np.r_[d * 0.68 * L + side * rng.uniform(-0.08, 0.08) * L, -0.06 * L],
                start + np.r_[d * L, -0.01 * L]]
        pts = _spline(ctrl, 16)
        rr = 0.42 * r_start * (1 - np.linspace(0, 1, len(pts))) ** 0.9 + 0.7
        parts.append((pts, rr))
        k = int(np.clip(L / 10, 4, 9))
        for j, s in enumerate(np.linspace(0.3, 0.96, k)):
            q, u = _along(pts, s)
            u2 = u[:2] / max(np.linalg.norm(u[:2]), 1e-9)
            sgn = 1 if j % 2 == 0 else -1
            l2 = (0.26 + 0.12 * rng.random()) * L * (1 - 0.4 * s) + 8
            twig(q, _turn(u2, sgn * rng.uniform(0.85, 1.2)), l2, max(0.45 * rr[int(s * (len(rr) - 1))], 0.8),
                 parts, 0, rng.uniform(0.02, 0.12))
        return parts, foliage(parts, 1)

    def fits(tufts):
        tips = _tips(tufts)
        return bool(np.all(shapely.contains_xy(canopy, tips[:, 0], tips[:, 1]))) and tips[:, 2].max() <= z0 + H

    # ---- primary branches: golden-angle spiral, lowest longest, no front branch low down
    heights = np.array([0.37, 0.43, 0.49, 0.55, 0.61, 0.67, 0.73, 0.79, 0.85]) + rng.uniform(-0.015, 0.015, 9)
    theta = math.radians(170.0)
    for i, tb in enumerate(heights):
        p, rt = trunk_at(tb)
        th = theta + i * GOLDEN + rng.uniform(-0.2, 0.2)
        d = np.array([math.cos(th), math.sin(th)])
        if i < 3 and d @ np.array([0.0, -1.0]) > math.cos(math.radians(40)):
            d = _turn(d, math.radians(65) * (1 if d[0] >= 0 else -1))
        L = max(reach(p, d) * (0.98 - 0.8 * (tb - 0.37)), 25.0)
        for _ in range(14):                                   # shorten until the foliage fits
            parts, tufts = branch(p, d, L, rt)
            if fits(tufts):
                break
            L *= 0.95
        else:
            continue                                          # no room for this branch
        tree_out.branches += parts
        tree_out.tufts += tufts

    # ---- the apex: shoots rising from the top of the trunk, a dome of foliage
    top, rt = trunk_at(1.0)
    for scale in (1.0, 0.85, 0.7, 0.55, 0.4):
        parts = []
        for k in range(6):
            th = 2 * np.pi * k / 6 + rng.uniform(-0.3, 0.3)
            twig(top, np.array([math.cos(th), math.sin(th)]), 0.1 * H * scale, max(rt * 0.6, 1.0), parts, 1,
                 rng.uniform(0.35, 0.6))
        tufts = foliage(parts, 0, NEEDLE * 0.9)
        if fits(tufts):
            tree_out.branches += parts
            tree_out.tufts += tufts
            break
    return tree_out


def tuft_mesh(n: int = 44, seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """One pine shoot, size 1 along +z: a short stem with needles in fascicles all along it,
    radiating forwards and out like a bottle brush. Scale it by the needle length."""
    v, f, _ = _shoot(n, seed)
    return v, f


def _shoot(n: int, seed: int):
    rng = np.random.default_rng(seed)
    verts, faces, tips = [], [], []
    stem = 0.45
    for k in range(n):
        at = stem * (k / n) ** 0.7                      # denser towards the tip of the shoot
        phi = math.radians(rng.uniform(25, 65) * (1 - 0.3 * at / stem))
        az = k * 2.39996 + rng.uniform(-0.2, 0.2)
        L = rng.uniform(0.62, 0.9)
        d = np.array([math.sin(phi) * math.cos(az), math.sin(phi) * math.sin(az), math.cos(phi)])
        out = np.array([math.cos(az), math.sin(az), 0.0])
        side = np.cross(d, out)
        side /= np.linalg.norm(side)
        base = np.array([0.0, 0.0, at])
        i0 = len(verts)
        for s in (0.0, 1 / 3, 2 / 3, 1.0):
            c = base + d * L * s + out * 0.06 * L * s * s
            w = 0.022 * (1 - s) + 0.004
            verts += [c - side * w, c + side * w]
        for j in range(3):
            a = i0 + 2 * j
            faces += [(a, a + 1, a + 3), (a, a + 3, a + 2)]
        tips += [i0 + 6, i0 + 7]
    i0 = len(verts)                                     # the stem itself
    verts += [[-0.02, 0, 0], [0.02, 0, 0], [0.012, 0, stem + 0.05], [-0.012, 0, stem + 0.05]]
    faces += [(i0, i0 + 1, i0 + 2), (i0, i0 + 2, i0 + 3)]
    tips += [i0 + 2, i0 + 3]
    verts = np.array(verts)
    return verts, np.array(faces), verts[tips]


SHOOT_TIPS = _shoot(44, 3)[2]         # the needle tips of the unit shoot tuft_mesh() draws
