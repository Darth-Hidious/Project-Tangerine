"""The ground of the garden outside the raked sand: moss, the hill, the stream bed and its banks.

One function, ``terrain``, gives for every cell of a grid over the tray: what it is (a zone code),
the height of the ground (for water cells, the bed under the water) and the water surface. The
Python renderer and the 3D viewer's export both use it, so they show the same garden.

Heights are mm above the sand surface. Water levels come from the stream's ``levels``: a reach
slopes towards its lip, a plunge pool below a cascade sits at the level after the drop, the pool
at its own height. The ground never comes closer than BANK above the water it borders, so no
water stands above its banks.

The frame is one height all round. At the walls the ground stays RIM_FREEBOARD below its rim and
climbs away from them no faster than RIM_SLOPE, so soil and water stay inside the box; the hill and
the spring rise above the rim only further in, as the mounds of a saikei tray do.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import shapely
from scipy.ndimage import distance_transform_edt, gaussian_filter, maximum_filter
from shapely.geometry import LineString, MultiLineString, Point, Polygon

from .config import Garden
from .geometry import sand_region, stone_polygons
from .water import cascades, living_moss_zone, stream_profile

SAND, WATER, LIVING, PRESERVED, ROCK, BRIDGE, STONE, KERB, OUTSIDE = range(9)
BANK = 5.0            # ground at the water's edge stands this much above the water (mm)
LIP = 4.0             # ... over this width from the edge (mm)
BANK_FALL = 0.5       # beyond the lip a bank falls away no faster than this (rise over run)
KERB_WIDTH, KERB_HEIGHT = 5.0, 3.0     # the low rim of the sealed sand basin
SAND_SLOPE = 1.0      # the ground climbs away from the kerb no faster than this: no cliff at the sand
VALLEY = 0.6          # the ground climbs away from water no faster than this (rise over run) ...
STEEP = 1.2           # ... unless a higher bank beside it needs more; even then never faster than this
RIM_FREEBOARD = 5.0   # at the walls the ground stays this far below the rim (mm)
RIM_SLOPE = 0.8       # and climbs away from them no faster than this
CORNER = 12.0         # mm: how far the ground rounds off into the tray's corners
SMOOTH = 3.0          # mm: the land is rounded off at this scale (before the moss cushions go on)
CUSHION = 3.4         # mm: the most the moss cushions (the fine noise) add to the ground
DEPTH = {"reach": 8.0, "plunge": 22.0, "pool": 25.0}   # water depth (mm) over the bed
HILL_RAMP = 90.0      # the hill reaches full height this far inside its outline


@dataclass
class Terrain:
    x: np.ndarray            # cell centres (mm), 1-D
    y: np.ndarray
    zone: np.ndarray         # (ny, nx) uint8 zone codes
    ground: np.ndarray       # (ny, nx) ground (or water-bed) height above the sand surface
    water: np.ndarray        # (ny, nx) water surface above the sand surface, NaN where dry


def _noise(shape, sigma_cells, seed):
    rng = np.random.default_rng(seed)
    n = gaussian_filter(rng.standard_normal(shape), sigma_cells)
    return n / (np.abs(n).max() + 1e-12)


def _smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _along_wall(a, b, W: float, D: float, tol: float = 1.0) -> bool:
    """Whether the segment a-b runs along one of the tray's walls."""
    return any(abs(a[k] - v) < tol and abs(b[k] - v) < tol for k, v in ((0, 0.0), (0, W), (1, 0.0), (1, D)))


def rim_cap(garden: Garden, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Highest the ground may stand at X, Y (mm above the sand) and stay inside the frame. The
    distance to the walls is a soft minimum, never more than the true one, so the ground rounds
    off into the corners instead of meeting them in a crease."""
    W, D = garden.tray.width, garden.tray.depth
    rim = garden.tray.wall_height - garden.tray.bed_depth
    k = CORNER
    d = np.stack([X, W - X, Y, D - Y])
    d_wall = d.min(axis=0) - k * np.log(np.exp(-(d - d.min(axis=0)) / k).sum(axis=0))
    return rim - RIM_FREEBOARD + RIM_SLOPE * np.maximum(d_wall, 0.0)


def bank_heights(garden: Garden, t: "Terrain") -> np.ndarray:
    """What holds the water back at each cell: the ground, or, where a rock stands, the rock's top.
    Rocks at the water's edge are set into the liner and sealed to it, as in any built pond."""
    X, Y = np.meshgrid(t.x, t.y)
    h = t.ground.copy()
    for f in garden.features:
        if f.kind == "rock":
            m = shapely.contains_xy(Polygon(f.outline), X, Y)
            h[m] = np.maximum(h[m], f.height)
    return h


def water_levels(garden: Garden, X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Water surface (NaN where dry) and water depth for the points X, Y."""
    level = np.full(X.shape, np.nan)
    depth = np.zeros(X.shape)
    pts = shapely.points(X.ravel(), Y.ravel())
    for f in garden.features:
        if f.kind == "stream" and f.levels:
            reaches, _ = stream_profile(f)
            inside = shapely.contains(LineString(f.outline).buffer(f.width / 2), pts).reshape(X.shape)
            best = np.full(X.shape, np.inf)
            for r in reaches:
                a, b = np.asarray(f.outline[r["from"]]), np.asarray(f.outline[r["from"] + 1])
                ab = b - a
                t = np.clip(((X - a[0]) * ab[0] + (Y - a[1]) * ab[1]) / (ab @ ab), 0.0, 1.0)
                d = np.hypot(X - (a[0] + t * ab[0]), Y - (a[1] + t * ab[1]))
                lv = f.levels[r["from"]] - r["fall_mm"] * t
                take = inside & (d < best)
                level[take] = lv[take]
                depth[take] = DEPTH["reach"]
                best = np.where(take, d, best)
            for i in cascades(f):
                if f.plunge > 0:
                    m = shapely.contains(Point(f.outline[i]).buffer(f.plunge), pts).reshape(X.shape)
                    level[m] = f.levels[i]
                    depth[m] = DEPTH["plunge"]
    for f in garden.features:
        if f.kind == "pool":
            m = shapely.contains(Polygon(f.outline), pts).reshape(X.shape)
            level[m] = f.height
            depth[m] = DEPTH["pool"]
    return level, depth


def terrain(garden: Garden, dx: float = 2.0, seed: int = 11, overlays: bool = True) -> Terrain:
    """``overlays`` raises the rocks, the bridge deck and the stones into the ground (for the
    heightfield renderer); without them they are only marked in ``zone`` (the 3D viewer
    draws them as meshes)."""
    W, D = garden.tray.width, garden.tray.depth
    x = (np.arange(int(round(W / dx))) + 0.5) * dx
    y = (np.arange(int(round(D / dx))) + 0.5) * dx
    X, Y = np.meshgrid(x, y)
    shape = X.shape
    pts = shapely.points(X.ravel(), Y.ravel())
    sand_poly = sand_region(garden)
    sand = shapely.contains(sand_poly, pts).reshape(shape)
    level, depth = water_levels(garden, X, Y)
    water = ~np.isnan(level) & ~sand
    land = ~sand & ~water

    # ground: rises gently away from the sand, plus the hill, plus slow undulations
    cap = rim_cap(garden, X, Y)
    d_sand = distance_transform_edt(~sand) * dx
    ground = np.minimum(0.35 * d_sand, 24.0)
    for f in garden.features:
        if f.kind == "hill":
            # the hill climbs from its open edges; where it meets the walls the rim rule shapes it
            poly = Polygon(f.outline)
            inside = shapely.contains(poly, pts).reshape(shape)
            ring = list(f.outline) + [f.outline[0]]
            edges = [(a, b) for a, b in zip(ring, ring[1:]) if not _along_wall(a, b, W, D)]
            rise_from = MultiLineString(edges) if edges else poly.exterior
            d_in = np.where(inside, shapely.distance(rise_from, pts).reshape(shape), 0.0)
            ground += f.height * _smoothstep(d_in / HILL_RAMP)
    ground += 3.0 * _noise(shape, 25.0 / dx, seed + 3)
    ground = np.where(land, np.minimum(ground, cap - CUSHION), ground)

    # banks and valleys. Every water surface raises a bank round itself that falls away no faster
    # than BANK_FALL, and caps the ground beside it to climb no faster than VALLEY. Taken over all
    # the water at once (in 1 mm steps of level) both are continuous, so no cliff forms where the
    # nearest water changes from one reach to a lower one.
    lip = np.zeros(shape, bool)
    if water.any():
        q = np.round(np.where(water, level, np.nan))
        bank, valley, steep = np.full(shape, -np.inf), np.full(shape, np.inf), np.full(shape, np.inf)
        for L in np.unique(q[water]):
            d = distance_transform_edt(q != L) * dx
            bank = np.maximum(bank, L + BANK - BANK_FALL * np.maximum(d - LIP, 0.0))
            valley = np.minimum(valley, L + BANK + VALLEY * d)
            steep = np.minimum(steep, L + BANK + STEEP * np.maximum(d - dx, 0.0))
        # the lip holds, exactly, the highest water within 6 mm: beside a cascade both levels are near
        d_water, (iw, jw) = distance_transform_edt(~water, return_indices=True)
        highest = maximum_filter(np.where(water, level, -np.inf), size=2 * int(round(6.0 / dx)) + 1)
        near_level = np.maximum(level[iw, jw], highest)
        lip = land & (d_water * dx <= LIP)
        # where a high bank meets lower water, the drop becomes a steep slope rather than a cliff
        # (the lips are put back below, so a bank too close to lower water still holds its own)
        ground = np.where(land, np.minimum(np.maximum(np.minimum(ground, valley), bank), steep), ground)
    # and where a bank's shoulder reaches the sand, it comes down to the kerb as a slope
    ground = np.where(land, np.minimum(ground, KERB_HEIGHT + SAND_SLOPE * np.maximum(d_sand - KERB_WIDTH, 0.0)), ground)

    # round off the creases where those rules meet (smoothing the land only), add the moss
    # cushions, then put back exactly what must hold: the lip of every bank and the rim
    sigma = SMOOTH / dx
    weight = gaussian_filter(land.astype(float), sigma)
    smooth = gaussian_filter(np.where(land, ground, 0.0), sigma) / np.maximum(weight, 1e-9)
    ground = np.where(land, smooth, ground)
    ground += 1.4 * _noise(shape, 2.0 / dx, seed + 2) + 2.0 * np.abs(_noise(shape, 3.0 / dx, seed + 12))  # <= CUSHION
    if water.any():
        ground = np.where(lip, np.maximum(ground, near_level + BANK), ground)
        d_in = distance_transform_edt(water) * dx                # a rounded bed, deepest mid-channel
        ground = np.where(water, level - depth * _smoothstep(d_in / 8.0), ground)

    # the walls: nothing on the land may stand above the rim at the wall (the banks must fit too)
    ground = np.where(land, np.minimum(ground, cap), ground)

    # the sand basin's kerb, then the sand surface itself at 0
    kerb = land & (d_sand <= KERB_WIDTH)
    ground = np.where(kerb, np.maximum(ground, KERB_HEIGHT), ground)
    ground = np.where(sand, 0.0, ground)

    zone = np.full(shape, PRESERVED, np.uint8)
    living = shapely.contains(living_moss_zone(garden), pts).reshape(shape) & land
    zone[living] = LIVING
    zone[kerb] = KERB
    zone[water] = WATER
    zone[sand] = SAND

    # rocks, the bridge and the stones sit on top
    def dome(mask, top, lo):
        d = distance_transform_edt(mask) * dx
        return lo + (top - lo) * (d / max(d.max(), 1e-9)) ** 0.5
    for f in garden.features:
        if f.kind == "rock":
            m = shapely.contains(Polygon(f.outline), pts).reshape(shape)
            if overlays and m.any():
                lo = float(np.nanmin(np.where(m, ground, np.nan)))
                ground = np.where(m, np.maximum(ground, dome(m, f.height, lo)), ground)
            zone[m] = ROCK
        elif f.kind == "bridge":
            poly = Polygon(f.outline)
            m = shapely.contains(poly, pts).reshape(shape)
            if overlays:
                c = np.asarray(poly.centroid.coords[0])
                half = max(np.hypot(*(np.asarray(poly.exterior.coords) - c).T))
                r = np.hypot(X - c[0], Y - c[1]) / half
                deck = f.height - 6.0 + 8.0 * np.sqrt(np.clip(1 - r**2, 0, 1))
                ground = np.where(m, np.maximum(ground, deck), ground)
                zone[m] = BRIDGE
    for stone, poly in zip(garden.stones, stone_polygons(garden)):
        m = shapely.contains(poly, pts).reshape(shape)
        if overlays:
            top = stone.height - garden.tray.bed_depth
            ground = np.where(m, dome(m, top, -4.0) + 1.5 * _noise(shape, 1.5 / dx, seed + 6), ground)
        zone[m] = STONE
    return Terrain(x, y, zone, ground, np.where(water, level, np.nan))
