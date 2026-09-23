"""The ground of the garden outside the raked sand: moss, the hill, the stream bed and its banks.

One function, ``terrain``, gives for every cell of a grid over the tray: what it is (a zone code),
the height of the ground (for water cells, the bed under the water) and the water surface. The
Python renderer and the 3D viewer's export both use it, so they show the same garden.

Heights are mm above the sand surface. Water levels come from the stream's ``levels``: a reach
slopes towards its lip, a plunge pool below a cascade sits at the level after the drop, the pool
at its own height. The ground never comes closer than BANK above the water it borders, so no
water stands above its banks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import shapely
from scipy.ndimage import distance_transform_edt, gaussian_filter, maximum_filter
from shapely.geometry import LineString, Point, Polygon

from .config import Garden
from .geometry import sand_region, stone_polygons
from .water import cascades, living_moss_zone, stream_profile

SAND, WATER, LIVING, PRESERVED, ROCK, BRIDGE, STONE, KERB, OUTSIDE = range(9)
BANK = 5.0            # ground at the water's edge stands this much above the water (mm)
KERB_WIDTH, KERB_HEIGHT = 5.0, 3.0     # the low rim of the sealed sand basin
VALLEY = 0.9          # steepest slope (rise over run) of the ground climbing away from water
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

    # ground: rises gently away from the sand, plus the hill, plus moss cushions
    d_sand = distance_transform_edt(~sand) * dx
    ground = np.minimum(0.35 * d_sand, 24.0)
    for f in garden.features:
        if f.kind == "hill":
            poly = Polygon(f.outline)
            inside = shapely.contains(poly, pts).reshape(shape)
            d_in = np.where(inside, shapely.distance(poly.exterior, pts).reshape(shape), 0.0)
            ground += f.height * _smoothstep(d_in / HILL_RAMP)
    ground += 1.4 * _noise(shape, 2.0 / dx, seed + 2) + 3.0 * _noise(shape, 25.0 / dx, seed + 3)
    ground += 2.0 * np.abs(_noise(shape, 3.0 / dx, seed + 12))

    # banks: never below the water they border, sloping down to the general ground away from it
    if water.any():
        d_water, (iw, jw) = distance_transform_edt(~water, return_indices=True)
        d_water *= dx
        # the highest water close by, not just the nearest: beside a cascade both levels are near
        highest = maximum_filter(np.where(water, level, -np.inf), size=2 * int(round(6.0 / dx)) + 1)
        near_level = np.maximum(level[iw, jw], highest)
        bank = near_level + BANK - 0.2 * np.maximum(d_water - 6.0, 0.0)
        valley = near_level + BANK + VALLEY * d_water            # the ground may not climb faster than this
        ground = np.where(land, np.maximum(np.minimum(ground, valley), bank), ground)
        d_in = distance_transform_edt(water) * dx                # a rounded bed, deepest mid-channel
        ground = np.where(water, level - depth * _smoothstep(d_in / 8.0), ground)

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
