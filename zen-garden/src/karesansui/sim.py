"""Heightfield model of the gravel bed and of the tools that shape it.

The bed is a height map ``h[i, j]`` (mm above the base plate, cell centre at
((j + 0.5) dx, (i + 0.5) dx)); stone cells are masked out of the grit. Four processes:

1. Tine carving. Each tine is a disc of radius tine_width / 2 that removes every grain
   above its tip and drops it in a ring just outside, biased sideways and slightly forwards
   (a wedge tine parts grit to both sides; almost nothing falls in behind it).
2. Avalanching. Wherever the height step to a neighbour exceeds tan(repose) x distance,
   grit slides down until it no longer does. Each move is a pairwise transfer that makes
   that one step exactly critical (a projection onto the slope constraint), cycled over the
   8-neighbourhood with alternating sweep direction. Mass is conserved to rounding error.
3. Screeding. A blade at a fixed height cuts everything above it into a berm carried in
   front of it and fills hollows below it from that berm; what is left is dropped where the
   blade lifts off.
4. The skid. A flat shoe ahead of the tines rests on the highest grains under it (a high
   percentile of the heights). On a floating head it sets the tine depth; when the head is
   held by a stop (or is rigid) it digs in and bulldozes whatever stands above its sole.

This is the cellular sand model used in computer graphics (Sumner, O'Brien & Hodgins 1999,
"Animating sand, mud, and snow") with an exactly mass-conserving relaxation. It resolves
geometry, mass balance and slope limits at the cell size. It knows nothing about grain size,
grain shape beyond the repose angle, compaction or dilatancy: those need Phase 0 tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import shapely
from numba import njit

from .config import Garden
from .geometry import stone_polygons


@dataclass(frozen=True)
class SimCfg:
    """Numerical and model parameters. These are modelling choices, not measurements."""
    dx: float = 1.0                  # cell size, mm
    step: float = 0.5                # tool advance per update, in cells
    lateral_bias: float = 1.0        # deposit weight beside a tine ...
    forward_bias: float = 0.3        # ... and ahead of it (nothing falls in behind)
    ring_width: float = 2.0          # width of the deposit ring around a tine, mm
    skid_percentile: float = 95.0    # the skid rests on this percentile of the heights under it
    berm_mix: float = 0.05           # share of the berm's lateral imbalance evened out per update
                                     # (loose grit slumps across the blade within ~1 cm of travel)
    local_iters: int = 10            # relaxation sweeps around each tool after every update (converged, see tests)
    relax_tol: float = 0.005         # mm of slope excess tolerated at convergence
    relax_max_iter: int = 2000


# ============================================================================ numba kernels
_DI = np.array([-1, -1, -1, 0, 0, 1, 1, 1], dtype=np.int64)
_DJ = np.array([-1, 0, 1, -1, 1, -1, 0, 1], dtype=np.int64)


# Pair directions (east, north, north-east, north-west) and the index whose parity colours them.
_PDI = np.array([0, 1, 1, 1], dtype=np.int64)
_PDJ = np.array([1, 0, 1, -1], dtype=np.int64)


@njit(cache=True)
def _relax(h, bed, i0, i1, j0, j1, t4, t8, max_iter, tol):
    """Avalanche until no step inside [i0:i1, j0:j1] exceeds its threshold. Returns (iters, worst).

    Each sub-step handles one direction and one parity colour, so the pairs it touches are
    disjoint: no grain can cascade along the sweep. Which direction and colour go first is
    drawn per iteration from a hash of the window and iteration number (deterministic), so
    the residual order dependence has no preferred side instead of a systematic one."""
    ny, nx = h.shape
    i0 = max(i0, 0)
    j0 = max(j0, 0)
    i1 = min(i1, ny)
    j1 = min(j1, nx)
    worst = 0.0
    for it in range(max_iter):
        worst = 0.0
        seed = (i0 * 73856093) ^ (j0 * 19349663) ^ ((it + 1) * 83492791)
        seed = (seed * 2654435761) & 0xFFFFFFFF
        rev = (seed >> 11) & 1
        swap_diag = (seed >> 17) & 1
        col0 = (seed >> 23) & 1
        for s in range(8):
            k = s // 2 if rev == 0 else 3 - s // 2
            if swap_diag == 1 and k >= 2:
                k = 5 - k
            colour = (s + col0) % 2
            di = _PDI[k]
            dj = _PDJ[k]
            thr = t8 if (di != 0 and dj != 0) else t4
            for i in range(i0, i1):
                ni = i + di
                if ni < 0 or ni >= ny:
                    continue
                for j in range(j0, j1):
                    par = j if k == 0 else i
                    if par % 2 != colour:
                        continue
                    nj = j + dj
                    if nj < 0 or nj >= nx:
                        continue
                    if not bed[i, j] or not bed[ni, nj]:
                        continue
                    d = h[i, j] - h[ni, nj]
                    if d > thr + tol:
                        q = 0.5 * (d - thr)
                        h[i, j] -= q
                        h[ni, nj] += q
                        if d - thr > worst:
                            worst = d - thr
                    elif -d > thr + tol:
                        q = 0.5 * (-d - thr)
                        h[i, j] += q
                        h[ni, nj] -= q
                        if -d - thr > worst:
                            worst = -d - thr
        if worst <= tol:
            return it + 1, worst
    return max_iter, worst


@njit(cache=True)
def _carve(h, bed, cx, cy, z_tip, ux, uy, r_t, r_ring, lat, fwd, dx):
    """Cut a tine disc down to z_tip and drop the grit on a ring around it. Returns volume moved (mm^3)."""
    ny, nx = h.shape
    i0 = max(int((cy - r_ring) / dx) - 1, 0)
    i1 = min(int((cy + r_ring) / dx) + 2, ny)
    j0 = max(int((cx - r_ring) / dx) - 1, 0)
    j1 = min(int((cx + r_ring) / dx) + 2, nx)
    removed = 0.0
    wsum = 0.0
    for i in range(i0, i1):
        yc = (i + 0.5) * dx
        for j in range(j0, j1):
            if not bed[i, j]:
                continue
            ox = (j + 0.5) * dx - cx
            oy = yc - cy
            d = math.sqrt(ox * ox + oy * oy)
            if d <= r_t:
                if h[i, j] > z_tip:
                    removed += h[i, j] - z_tip
                    h[i, j] = z_tip
            elif d <= r_ring:
                c = (ox * ux + oy * uy) / d
                s = abs(ox * uy - oy * ux) / d
                w = lat * s + fwd * c
                if w > 0.0:
                    wsum += w
    if removed == 0.0:
        return 0.0
    if wsum == 0.0:
        # Nowhere to put it (walled in by stone): give it back evenly to the disc.
        n = 0
        for i in range(i0, i1):
            for j in range(j0, j1):
                if bed[i, j]:
                    ox = (j + 0.5) * dx - cx
                    oy = (i + 0.5) * dx - cy
                    if ox * ox + oy * oy <= r_t * r_t:
                        n += 1
        for i in range(i0, i1):
            for j in range(j0, j1):
                if bed[i, j]:
                    ox = (j + 0.5) * dx - cx
                    oy = (i + 0.5) * dx - cy
                    if ox * ox + oy * oy <= r_t * r_t:
                        h[i, j] += removed / n
        return 0.0
    for i in range(i0, i1):
        yc = (i + 0.5) * dx
        for j in range(j0, j1):
            if not bed[i, j]:
                continue
            ox = (j + 0.5) * dx - cx
            oy = yc - cy
            d = math.sqrt(ox * ox + oy * oy)
            if r_t < d <= r_ring:
                c = (ox * ux + oy * uy) / d
                s = abs(ox * uy - oy * ux) / d
                w = lat * s + fwd * c
                if w > 0.0:
                    h[i, j] += removed * w / wsum
    return removed * dx * dx


@njit(cache=True)
def _skid(h, bed, cx, cy, ux, uy, gap, length, half_w, pct, lo, hi, dx):
    """Sole height of the skid: it rests on the grit (a high percentile of the heights under it,
    since a rigid plate is carried by the highest grains), clamped to [lo, hi], the head's stops.

    Only when a stop holds the sole below its rest height does the skid dig in: then it presses
    everything above the sole into the hollows beneath it and pushes the rest out sideways.
    (Pressing while floating would flatten the cells the next rest height is read from and
    pin the skid at a constant height - an artefact, not physics.)"""
    ny, nx = h.shape
    vx, vy = -uy, ux
    # bounding box of the skid rectangle
    xs = np.empty(4)
    ys = np.empty(4)
    k = 0
    for a in (gap, gap + length):
        for b in (-half_w, half_w):
            xs[k] = cx + a * ux + b * vx
            ys[k] = cy + a * uy + b * vy
            k += 1
    i0 = max(int(ys.min() / dx) - 1, 0)
    i1 = min(int(ys.max() / dx) + 2, ny)
    j0 = max(int(xs.min() / dx) - 1, 0)
    j1 = min(int(xs.max() / dx) + 2, nx)
    buf = np.empty((i1 - i0) * (j1 - j0))
    n = 0
    for i in range(i0, i1):
        for j in range(j0, j1):
            if not bed[i, j]:
                continue
            ox = (j + 0.5) * dx - cx
            oy = (i + 0.5) * dx - cy
            u = ox * ux + oy * uy
            v = ox * vx + oy * vy
            if gap <= u <= gap + length and -half_w <= v <= half_w:
                buf[n] = h[i, j]
                n += 1
    if n == 0:
        return hi
    rest = np.percentile(buf[:n], pct)
    sole = min(max(rest, lo), hi)
    if rest <= sole + 1e-9:
        return sole
    excess = 0.0
    deficit = 0.0
    for i in range(i0, i1):
        for j in range(j0, j1):
            if not bed[i, j]:
                continue
            ox = (j + 0.5) * dx - cx
            oy = (i + 0.5) * dx - cy
            u = ox * ux + oy * uy
            v = ox * vx + oy * vy
            if gap <= u <= gap + length and -half_w <= v <= half_w:
                if h[i, j] > sole:
                    excess += h[i, j] - sole
                else:
                    deficit += sole - h[i, j]
    if excess == 0.0:
        return sole
    n_edge = 0
    for i in range(i0, i1):
        for j in range(j0, j1):
            if bed[i, j]:
                ox = (j + 0.5) * dx - cx
                oy = (i + 0.5) * dx - cy
                u = ox * ux + oy * uy
                v = ox * vx + oy * vy
                if gap <= u <= gap + length and half_w < abs(v) <= half_w + 2.0 * dx:
                    n_edge += 1
    fill = min(excess, deficit)
    spill = excess - fill if n_edge > 0 else 0.0
    cut = (fill + spill) / excess          # < 1 only when the grit has nowhere to go
    for i in range(i0, i1):
        for j in range(j0, j1):
            if not bed[i, j]:
                continue
            ox = (j + 0.5) * dx - cx
            oy = (i + 0.5) * dx - cy
            u = ox * ux + oy * uy
            v = ox * vx + oy * vy
            if gap <= u <= gap + length and -half_w <= v <= half_w:
                if h[i, j] > sole:
                    h[i, j] -= (h[i, j] - sole) * cut
                elif deficit > 0.0:
                    h[i, j] += (sole - h[i, j]) * fill / deficit
    if spill > 0.0:
        for i in range(i0, i1):
            for j in range(j0, j1):
                if not bed[i, j]:
                    continue
                ox = (j + 0.5) * dx - cx
                oy = (i + 0.5) * dx - cy
                u = ox * ux + oy * uy
                v = ox * vx + oy * vy
                if gap <= u <= gap + length and half_w < abs(v) <= half_w + 2.0 * dx:
                    h[i, j] += spill / n_edge
    return sole


@njit(cache=True)
def _cell(px, py, dx, ny, nx):
    """Grid cell containing (px, py), or (-1, -1) outside the grid (floor, not truncation)."""
    if px < 0.0 or py < 0.0:
        return -1, -1
    i = int(py / dx)
    j = int(px / dx)
    if i >= ny or j >= nx:
        return -1, -1
    return i, j


@njit(cache=True)
def _blade(h, bed, cx, cy, ux, uy, half_w, z, carry, dx, mix):
    """One screed update: cut above z into the berm, fill below z from it. carry has one bin per sample.

    Bins that sit over a wall or a stone hold no grit: whatever the berm would spread into
    them stays in the nearest bin over the bed, so no volume can leave the tray."""
    ny, nx = h.shape
    vx, vy = -uy, ux
    nb = carry.shape[0]
    valid = np.zeros(nb, dtype=np.bool_)
    moved = 0.0
    for b in range(nb):
        s = -half_w + (b + 0.5) * (2.0 * half_w / nb)
        i, j = _cell(cx + s * vx, cy + s * vy, dx, ny, nx)
        if i < 0 or not bed[i, j]:
            continue
        valid[b] = True
        if h[i, j] > z:
            carry[b] += h[i, j] - z
            moved += h[i, j] - z
            h[i, j] = z
        elif carry[b] > 0.0:
            f = min(z - h[i, j], carry[b])
            h[i, j] += f
            carry[b] -= f
    # Grit carried by a bin that has just moved off the bed goes to its nearest valid bin.
    for b in range(nb):
        if not valid[b] and carry[b] > 0.0:
            best = -1
            for r in range(1, nb):
                if b - r >= 0 and valid[b - r]:
                    best = b - r
                    break
                if b + r < nb and valid[b + r]:
                    best = b + r
                    break
            if best >= 0:
                carry[best] += carry[b]
                carry[b] = 0.0
    # The berm slumps sideways along the blade: move each valid bin a step towards the mean.
    total = 0.0
    n = 0
    for b in range(nb):
        if valid[b]:
            total += carry[b]
            n += 1
    if n > 0 and mix > 0.0:
        mean = total / n
        for b in range(nb):
            if valid[b]:
                carry[b] += mix * (mean - carry[b])
    return moved


@njit(cache=True)
def _dump(h, bed, cx, cy, ux, uy, half_w, carry, dx):
    """Drop the berm just ahead of the blade. A bin with no bed cell ahead hands its grit to the
    nearest bin that has one. Returns what could not be placed at all (0 unless the blade is off the bed)."""
    ny, nx = h.shape
    vx, vy = -uy, ux
    nb = carry.shape[0]
    ti = np.full(nb, -1, dtype=np.int64)
    tj = np.full(nb, -1, dtype=np.int64)
    for b in range(nb):
        s = -half_w + (b + 0.5) * (2.0 * half_w / nb)
        for ahead in (1.0, 2.0, 0.0, 3.0, -1.0):
            i, j = _cell(cx + ahead * dx * ux + s * vx, cy + ahead * dx * uy + s * vy, dx, ny, nx)
            if i >= 0 and bed[i, j]:
                ti[b] = i
                tj[b] = j
                break
    lost = 0.0
    for b in range(nb):
        if carry[b] == 0.0:
            continue
        t = b
        if ti[b] < 0:
            t = -1
            for r in range(1, nb):
                if b - r >= 0 and ti[b - r] >= 0:
                    t = b - r
                    break
                if b + r < nb and ti[b + r] >= 0:
                    t = b + r
                    break
        if t >= 0:
            h[ti[t], tj[t]] += carry[b]
        else:
            lost += carry[b]
        carry[b] = 0.0
    return lost


# ============================================================================ the bed
@dataclass
class PassStats:
    kind: str
    samples: int = 0
    moved_mm3: float = 0.0
    sole: list = field(default_factory=list)        # skid sole height per sample (rake only)
    tip: list = field(default_factory=list)         # tine tip height per sample (rake only)
    at_stop: int = 0                                # samples with the floating head at a stop


class Bed:
    def __init__(self, garden: Garden, cfg: SimCfg | None = None, unevenness: np.ndarray | None = None):
        self.garden, self.cfg = garden, cfg or SimCfg()
        dx = self.cfg.dx
        self.nx = int(round(garden.tray.width / dx))
        self.ny = int(round(garden.tray.depth / dx))
        xs = (np.arange(self.nx) + 0.5) * dx
        ys = (np.arange(self.ny) + 0.5) * dx
        self.X, self.Y = np.meshgrid(xs, ys)
        self.stone = np.zeros((self.ny, self.nx), bool)
        for poly in stone_polygons(garden):
            self.stone |= shapely.contains_xy(poly, self.X, self.Y)
        self.bed = ~self.stone
        self.h = np.full((self.ny, self.nx), garden.tray.bed_depth, dtype=np.float64)
        if unevenness is not None:
            self.h += unevenness
        self.h[self.stone] = 0.0
        phi = math.radians(garden.grit.repose_deg)
        self.t4 = math.tan(phi) * dx
        self.t8 = math.tan(phi) * dx * math.sqrt(2.0)
        self.lost_mm3 = 0.0                          # berm that could not be put back (should stay 0)
        self.history: list[PassStats] = []

    # ------------------------------------------------------------------ bookkeeping
    def volume(self) -> float:
        """Grit volume in mm^3 (heights above the base plate over bed cells)."""
        return float(self.h[self.bed].sum() * self.cfg.dx**2)

    def relax(self, bbox=None) -> tuple[int, float]:
        i0, i1, j0, j1 = bbox if bbox is not None else (0, self.ny, 0, self.nx)
        return _relax(self.h, self.bed, i0, i1, j0, j1, self.t4, self.t8, self.cfg.relax_max_iter, self.cfg.relax_tol)

    def max_slope_excess(self) -> float:
        """Largest step above the repose threshold anywhere in the bed (mm); 0 when relaxed."""
        h, bed = self.h, self.bed
        worst = 0.0
        for di, dj in ((0, 1), (1, 0), (1, 1), (1, -1)):
            thr = self.t8 if di and dj else self.t4
            a = h[max(di, 0):, max(dj, 0): self.nx + min(dj, 0)]
            b = h[: self.ny - di, max(-dj, 0): self.nx - max(dj, 0)]
            ma = bed[max(di, 0):, max(dj, 0): self.nx + min(dj, 0)]
            mb = bed[: self.ny - di, max(-dj, 0): self.nx - max(dj, 0)]
            both = ma & mb
            if both.any():
                worst = max(worst, float((np.abs(a - b)[both] - thr).max()))
        return max(worst, 0.0)

    def _bbox(self, xs, ys, margin):
        dx = self.cfg.dx
        return (int((ys.min() - margin) / dx), int((ys.max() + margin) / dx) + 1,
                int((xs.min() - margin) / dx), int((xs.max() + margin) / dx) + 1)

    # ------------------------------------------------------------------ tools
    def rake(self, x, y, heading, z, released=None, rigid: bool | None = None) -> PassStats:
        """Drag the rake head along samples (x, y, heading rad, z = carriage/blade height).

        The head geometry comes from the garden's Rake: skid sole at z - hang on the lower stop,
        up to float_travel higher when floating; tines ``depth`` below the sole."""
        g, c, dx = self.garden, self.cfg, self.cfg.dx
        rake = g.rake
        rigid = (rake.mode == "fixed") if rigid is None else rigid
        released = np.ones(len(x), bool) if released is None else released
        offsets = np.asarray(rake.tine_offsets)
        r_t, r_ring = rake.tine_width / 2, rake.tine_width / 2 + c.ring_width
        w = int(math.ceil((r_ring + 3 * dx) / dx))
        st = PassStats("rake")
        for k in range(len(x)):
            ux, uy = math.cos(heading[k]), math.sin(heading[k])
            lo = z[k] - rake.hang
            hi = lo + rake.float_travel
            if not released[k]:
                sole = hi
                if sole - rake.depth >= self.h_max_near(x[k], y[k]):
                    continue
            elif rigid:
                sole = _skid(self.h, self.bed, x[k], y[k], ux, uy, rake.skid_gap, rake.skid_length,
                             rake.skid_width / 2, c.skid_percentile, lo, lo, dx)
            else:
                sole = _skid(self.h, self.bed, x[k], y[k], ux, uy, rake.skid_gap, rake.skid_length,
                             rake.skid_width / 2, c.skid_percentile, lo, hi, dx)
                if sole <= lo + 1e-9 or sole >= hi - 1e-9:
                    st.at_stop += 1
            tip = sole - rake.depth
            vx, vy = -uy, ux
            for d in offsets:
                cx, cy = x[k] + d * vx, y[k] + d * vy
                st.moved_mm3 += _carve(self.h, self.bed, cx, cy, tip, ux, uy, r_t, r_ring,
                                       c.lateral_bias, c.forward_bias, dx)
                i, j = int(cy / dx), int(cx / dx)
                _relax(self.h, self.bed, i - w, i + w + 1, j - w, j + w + 1, self.t4, self.t8, c.local_iters, c.relax_tol)
            st.samples += 1
            st.sole.append(sole)
            st.tip.append(tip)
        if st.samples:
            margin = rake.tine_half + rake.reach_ahead + 20.0
            self.relax(self._bbox(np.asarray(x), np.asarray(y), margin))
        self.history.append(st)
        return st

    def h_max_near(self, x, y, r: float = 80.0) -> float:
        dx = self.cfg.dx
        i0, i1 = max(int((y - r) / dx), 0), min(int((y + r) / dx) + 1, self.ny)
        j0, j1 = max(int((x - r) / dx), 0), min(int((x + r) / dx) + 1, self.nx)
        return float(self.h[i0:i1, j0:j1].max()) if i1 > i0 and j1 > j0 else 0.0

    def screed(self, x, y, heading, z) -> PassStats:
        """Drag the blade (bottom at z) along the samples; drop the berm where it ends."""
        g, c, dx = self.garden, self.cfg, self.cfg.dx
        half_w = g.screed.width / 2
        nb = int(math.ceil(2 * half_w / (0.5 * dx)))
        carry = np.zeros(nb)
        st = PassStats("screed")
        w = int(math.ceil((half_w + 3 * dx) / dx))
        for k in range(len(x)):
            ux, uy = math.cos(heading[k]), math.sin(heading[k])
            st.moved_mm3 += _blade(self.h, self.bed, x[k], y[k], ux, uy, half_w, z[k], carry, dx, c.berm_mix) * dx * dx
            st.samples += 1
            if k % 8 == 0:
                i, j = int(y[k] / dx), int(x[k] / dx)
                _relax(self.h, self.bed, i - w, i + w + 1, j - w, j + w + 1, self.t4, self.t8, 1, c.relax_tol)
        if len(x):
            ux, uy = math.cos(heading[-1]), math.sin(heading[-1])
            self.lost_mm3 += _dump(self.h, self.bed, x[-1], y[-1], ux, uy, half_w, carry, dx) * dx * dx
            self.relax(self._bbox(np.asarray(x), np.asarray(y), half_w + 30.0))
        self.history.append(st)
        return st

    # ------------------------------------------------------------------ views
    def surface(self) -> np.ndarray:
        """Grit heights with stones filled in by a rounded rock profile (for rendering)."""
        from scipy.ndimage import distance_transform_edt
        out = self.h.copy()
        for stone, poly in zip(self.garden.stones, stone_polygons(self.garden)):
            mask = shapely.contains_xy(poly, self.X, self.Y)
            if not mask.any():
                continue
            d = distance_transform_edt(mask) * self.cfg.dx
            prof = (d / d.max()) ** 0.45
            base = self.garden.tray.bed_depth - 5.0
            out[mask] = base + (stone.height - base) * prof[mask]
        return out


def unevenness_field(garden: Garden, amplitude: float, correlation: float, seed: int, dx: float = 1.0) -> np.ndarray:
    """Zero-mean smooth random height field with peak |value| = amplitude (mm), for sensitivity tests."""
    ny, nx = int(round(garden.tray.depth / dx)), int(round(garden.tray.width / dx))
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((ny, nx))
    ky = np.fft.fftfreq(ny, d=dx)[:, None]
    kx = np.fft.fftfreq(nx, d=dx)[None, :]
    filt = np.exp(-2 * (np.pi * correlation) ** 2 * (kx**2 + ky**2))
    field_ = np.real(np.fft.ifft2(np.fft.fft2(noise) * filt))
    field_ -= field_.mean()
    return field_ * (amplitude / np.abs(field_).max())
