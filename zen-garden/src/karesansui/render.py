"""Physically based (Lambertian) renders of the simulated garden.

Pipeline:
  1. build_scene: one height map for the whole unit (simulated sand, stones, moss terrain,
     water, rocks, bridge, lanterns, arm base, walnut frame, table) plus albedo per cell.
  2. bake: illuminance per cell from every lantern (a point source of ``lumens`` / 4 pi candela
     at its firebox) with ray-marched shadows, plus diffuse room light; radiance = albedo x E / pi.
     Diffuse shading does not depend on the viewpoint, so it is computed once.
  3. views: top-down (the baked map) and perspective (the height map rasterised as a mesh with a
     z-buffer, water given a specular glint per pixel).

Colours: lantern light from its correlated colour temperature via the Kang et al. (2002)
Planckian-locus fit, converted to linear sRGB; the camera is white-balanced to 3500 K so warm
light still reads warm.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import shapely
from numba import njit, prange
from scipy.ndimage import distance_transform_edt, gaussian_filter
from shapely.geometry import LineString, Point, Polygon

from .config import Garden
from .geometry import sand_region, stone_polygons

# ---------------------------------------------------------------------------------- colour
_XYZ_TO_SRGB = np.array([[3.2406, -1.5372, -0.4986],
                         [-0.9689, 1.8758, 0.0415],
                         [0.0557, -0.2040, 1.0570]])


def cct_to_xy(T: float) -> tuple[float, float]:
    """Kang et al. (2002) cubic fit to the Planckian locus, valid 1667-25000 K."""
    if not 1667 <= T <= 25000:
        raise ValueError("CCT outside 1667-25000 K")
    if T <= 4000:
        x = -0.2661239e9 / T**3 - 0.2343589e6 / T**2 + 0.8776956e3 / T + 0.179910
    else:
        x = -3.0258469e9 / T**3 + 2.1070379e6 / T**2 + 0.2226347e3 / T + 0.240390
    if T <= 2222:
        y = -1.1063814 * x**3 - 1.34811020 * x**2 + 2.18555832 * x - 0.20219683
    elif T <= 4000:
        y = -0.9549476 * x**3 - 1.37418593 * x**2 + 2.09137015 * x - 0.16748867
    else:
        y = 3.0817580 * x**3 - 5.87338670 * x**2 + 3.75112997 * x - 0.37001483
    return x, y


def cct_to_rgb(T: float, white: float = 6504.0) -> np.ndarray:
    """Linear sRGB of a light of colour temperature T and luminance 1, white-balanced so that a
    light at ``white`` K comes out neutral (von Kries scaling in linear RGB)."""
    def raw(t):
        x, y = cct_to_xy(t)
        xyz = np.array([x / y, 1.0, (1 - x - y) / y])
        return _XYZ_TO_SRGB @ xyz
    rgb = raw(T) / raw(white) * raw(white)[1]
    rgb = rgb / (0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2])     # unit luminance
    return np.clip(rgb, 0.0, None)


def to_srgb8(linear: np.ndarray) -> np.ndarray:
    a = np.clip(linear, 0.0, 1.0)
    s = np.where(a <= 0.0031308, 12.92 * a, 1.055 * np.power(a, 1 / 2.4) - 0.055)
    return (s * 255 + 0.5).astype(np.uint8)


# ---------------------------------------------------------------------------------- scene
@dataclass
class Scene:
    h: np.ndarray            # (ny, nx) mm above the table
    albedo: np.ndarray       # (ny, nx, 3) linear
    water: np.ndarray        # bool
    glow: np.ndarray         # (ny, nx) emissive luminance multiplier (lantern windows)
    x0: float                # world x of column 0's centre minus dx/2
    y0: float
    dx: float
    lights: list             # [(x, y, z, candela, rgb, skip radius)]
    base: float = 20.0       # tray floor above the table
    white: float = 2700.0    # camera white balance, K


def _noise(shape, sigma_cells, seed):
    rng = np.random.default_rng(seed)
    n = gaussian_filter(rng.standard_normal(shape), sigma_cells)
    return n / (np.abs(n).max() + 1e-12)


def build_scene(garden: Garden, sand_h: np.ndarray, dx: float, rim: float = 30.0, table: float = 60.0,
                seed: int = 11, white: float = 2700.0) -> Scene:
    """Height and albedo of the whole unit on a table. ``sand_h`` is the simulated bed (tray frame)."""
    W, D, S = garden.tray.width, garden.tray.depth, garden.tray.bed_depth
    base = 20.0                                    # tray floor above the table
    pad = rim + table
    nx, ny = int(round((W + 2 * pad) / dx)), int(round((D + 2 * pad) / dx))
    x0, y0 = -pad, -pad
    X = x0 + (np.arange(nx) + 0.5) * dx
    Y = y0 + (np.arange(ny) + 0.5) * dx
    XX, YY = np.meshgrid(X, Y)
    inside = (XX >= 0) & (XX < W) & (YY >= 0) & (YY < D)
    frame = (XX >= -rim) & (XX < W + rim) & (YY >= -rim) & (YY < D + rim) & ~inside
    h = np.zeros((ny, nx))
    alb = np.zeros((ny, nx, 3))
    alb[:] = (0.30, 0.22, 0.15)                    # oak table
    alb *= (1 + 0.08 * _noise((ny, nx), 3 / dx, seed))[..., None]
    h[frame] = base + garden.tray.wall_height
    walnut = np.array([0.14, 0.08, 0.045])
    alb[frame] = walnut * (1 + 0.1 * _noise((ny, nx), 2 / dx, seed + 1))[frame][:, None]

    # zones
    sand = shapely.contains_xy(sand_region(garden), XX, YY) & inside
    water = np.zeros_like(inside)
    for f in garden.features:
        if f.kind == "stream":
            water |= shapely.contains_xy(LineString(f.outline).buffer(f.width / 2), XX, YY)
        elif f.kind == "pool":
            water |= shapely.contains_xy(Polygon(f.outline), XX, YY)
    water &= inside & ~sand
    land = inside & ~sand & ~water

    # land: rises away from the sand, falls to the water; moss texture on top
    d_sand = distance_transform_edt(~sand) * dx
    d_water = distance_transform_edt(~water) * dx if water.any() else np.full_like(h, 1e3)
    z_land = np.minimum(S + np.minimum(0.35 * d_sand, 24.0), S - 4.0 + 0.55 * d_water)
    z_land += 1.4 * _noise((ny, nx), 2.0 / dx, seed + 2) + 5.0 * _noise((ny, nx), 25.0 / dx, seed + 3)
    z_land += 2.2 * np.abs(_noise((ny, nx), 3.0 / dx, seed + 12))       # cushions of moss
    h[land] = base + z_land[land]
    living = land & (d_water <= 45.0)
    tex = 1 + 0.35 * _noise((ny, nx), 1.2 / dx, seed + 4)
    alb[land & ~living] = np.array([0.12, 0.20, 0.055]) * tex[land & ~living][:, None]
    alb[living] = np.array([0.06, 0.15, 0.03]) * tex[living][:, None]

    # water surface
    h[water] = base + S - 7.0
    alb[water] = (0.035, 0.06, 0.065)

    # sand from the simulation (tray frame -> this grid)
    sh = sand_h
    j = np.clip(((XX - 0.0) / (W / sh.shape[1])).astype(int), 0, sh.shape[1] - 1)
    i = np.clip(((YY - 0.0) / (D / sh.shape[0])).astype(int), 0, sh.shape[0] - 1)
    h[sand] = base + sh[i[sand], j[sand]]
    grain = 1 + 0.07 * _noise((ny, nx), 0.35 / dx, seed + 5)
    alb[sand] = np.array([0.64, 0.61, 0.55]) * grain[sand][:, None]

    # stones, stream rocks, bridge
    def dome(mask, top, lo):
        d = distance_transform_edt(mask) * dx
        prof = (d / max(d.max(), 1e-9)) ** 0.5
        return lo + (top - lo) * prof
    for stone, poly in zip(garden.stones, stone_polygons(garden)):
        m = shapely.contains_xy(poly, XX, YY)
        h[m] = base + dome(m, stone.height, S - 4.0)[m] + 1.5 * _noise((ny, nx), 1.5 / dx, seed + 6)[m]
        alb[m] = np.array([0.075, 0.075, 0.08]) * (1 + 0.2 * _noise((ny, nx), 0.8 / dx, seed + 7))[m][:, None]
    for f in garden.features:
        if f.kind == "rock":
            m = shapely.contains_xy(Polygon(f.outline), XX, YY)
            h[m] = np.maximum(h[m], base + dome(m, S + f.height, S - 8.0)[m])
            alb[m] = np.array([0.20, 0.19, 0.17]) * (1 + 0.2 * _noise((ny, nx), 1 / dx, seed + 8))[m][:, None]
        elif f.kind == "bridge":
            poly = Polygon(f.outline)
            m = shapely.contains_xy(poly, XX, YY)
            c = np.array(poly.centroid.coords[0])
            half = max(np.hypot(*(np.asarray(poly.exterior.coords) - c).T))
            r = np.hypot(XX - c[0], YY - c[1]) / half
            h[m] = base + S + f.height - 6.0 + 8.0 * np.sqrt(np.clip(1 - r[m] ** 2, 0, 1))
            alb[m] = np.array([0.24, 0.15, 0.08]) * (1 + 0.15 * _noise((ny, nx), 0.6 / dx, seed + 9))[m][:, None]

    glow = np.zeros((ny, nx))
    lights = []
    rgb = cct_to_rgb
    for lan in garden.lanterns:
        rr = np.hypot(XX - lan.xy[0], YY - lan.xy[1])
        foot = rr <= lan.radius * 0.8
        h[foot] = np.maximum(h[foot], base + S + 0.35 * lan.height)       # base and post; the rest is a mesh
        alb[foot] = (0.30, 0.29, 0.27)
        lights.append((lan.xy[0], lan.xy[1], base + S + lan.light_height, lan.lumens / (4 * math.pi),
                       rgb(lan.cct, white=white), lan.radius * 1.3))    # skip the lantern's own body
    if garden.arm is not None:
        rr = np.hypot(XX - garden.arm.base[0], YY - garden.arm.base[1])
        m = rr <= garden.arm.base_radius
        h[m] = base + S + 60.0
        alb[m] = walnut
    return Scene(h, alb, water, glow, x0, y0, dx, lights, base, white)


# ---------------------------------------------------------------------------------- bake
@njit(cache=True, parallel=True)
def _bake(h, normals, lx, ly, lz, skip_r, x0, y0, dx, step):
    """Unshadowed-and-shadowed illuminance factor n.l / d^2 (1/mm^2) for one point light."""
    ny, nx = h.shape
    out = np.zeros((ny, nx))
    hmax = h.max()
    for i in prange(ny):
        py = y0 + (i + 0.5) * dx
        for j in range(nx):
            px = x0 + (j + 0.5) * dx
            pz = h[i, j]
            vx, vy, vz = lx - px, ly - py, lz - pz
            d2 = vx * vx + vy * vy + vz * vz
            d = math.sqrt(d2)
            ndl = (normals[i, j, 0] * vx + normals[i, j, 1] * vy + normals[i, j, 2] * vz) / d
            if ndl <= 0.0:
                continue
            # march towards the light; stop once inside the lantern's own body or above everything
            horiz = math.sqrt(vx * vx + vy * vy)
            n = int(horiz / step)
            lit = True
            for k in range(1, n):
                t = k * step / horiz
                qx = px + vx * t
                qy = py + vy * t
                qz = pz + vz * t + 0.3
                if (qx - lx) ** 2 + (qy - ly) ** 2 < skip_r * skip_r:
                    break
                if qz > hmax:
                    break
                jj = int((qx - x0) / dx)
                ii = int((qy - y0) / dx)
                if ii < 0 or jj < 0 or ii >= ny or jj >= nx:
                    break
                if h[ii, jj] > qz:
                    lit = False
                    break
            if lit:
                out[i, j] = ndl / d2
    return out


def normals_of(h: np.ndarray, dx: float) -> np.ndarray:
    gy, gx = np.gradient(h, dx)
    n = np.dstack([-gx, -gy, np.ones_like(h)])
    return n / np.linalg.norm(n, axis=2, keepdims=True)


@dataclass
class Baked:
    radiance: np.ndarray     # (ny, nx, 3) cd/m^2 per channel (linear)
    lux: np.ndarray          # (ny, nx) total illuminance
    normals: np.ndarray


def bake(scene: Scene, ambient_lux: float, ambient_cct: float = 3000.0, shadow_step: float | None = None) -> Baked:
    """Illuminance and radiance of every cell (lanterns with shadows, room light without)."""
    n = normals_of(scene.h, scene.dx)
    step = shadow_step or scene.dx
    E_rgb = np.zeros(scene.h.shape + (3,))
    lux = np.zeros(scene.h.shape)
    for (lx, ly, lz, cd, rgb, skip) in scene.lights:
        f = _bake(scene.h, n, lx, ly, lz, skip, scene.x0, scene.y0, scene.dx, step)
        e = cd * f * 1e6                                  # cd / mm^2 -> lux
        lux += e
        E_rgb += e[..., None] * rgb[None, None, :]
    amb = ambient_lux * (0.5 + 0.5 * n[..., 2])            # sky-like hemisphere, no occlusion
    lux += amb
    E_rgb += amb[..., None] * cct_to_rgb(ambient_cct, white=scene.white)[None, None, :]
    return Baked(scene.albedo * E_rgb / math.pi, lux, n)


# ---------------------------------------------------------------------------------- views
def expose(radiance: np.ndarray, key: float, reference: float) -> np.ndarray:
    """Map radiance so that ``reference`` cd/m^2 lands at display value ``key``, with a soft shoulder."""
    x = radiance * (key / reference)
    return x / (1 + 0.25 * x)


def topdown(baked: Baked, scene: Scene, reference: float, key: float = 0.5) -> np.ndarray:
    return to_srgb8(expose(baked.radiance, key, reference))[::-1]


@njit(cache=True)
def _raster(h, col, x0, y0, dx, cam, fwd, right, up, f, W, H, water, spec_l, spec_rgb, eye_spec):
    img = np.zeros((H, W, 3))
    zb = np.full((H, W), np.inf)
    ny, nx = h.shape
    for i in range(ny - 1):
        for j in range(nx - 1):
            # two triangles per cell quad
            for tri in range(2):
                if tri == 0:
                    ai, aj, bi, bj, ci, cj = i, j, i, j + 1, i + 1, j
                else:
                    ai, aj, bi, bj, ci, cj = i + 1, j + 1, i + 1, j, i, j + 1
                sx = np.empty(3)
                sy = np.empty(3)
                sz = np.empty(3)
                ok = True
                for v in range(3):
                    vi = ai if v == 0 else (bi if v == 1 else ci)
                    vj = aj if v == 0 else (bj if v == 1 else cj)
                    wx = x0 + (vj + 0.5) * dx - cam[0]
                    wy = y0 + (vi + 0.5) * dx - cam[1]
                    wz = h[vi, vj] - cam[2]
                    zc = wx * fwd[0] + wy * fwd[1] + wz * fwd[2]
                    if zc <= 1.0:
                        ok = False
                        break
                    xc = wx * right[0] + wy * right[1] + wz * right[2]
                    yc = wx * up[0] + wy * up[1] + wz * up[2]
                    sx[v] = W / 2 + f * xc / zc
                    sy[v] = H / 2 - f * yc / zc
                    sz[v] = zc
                if not ok:
                    continue
                minx = max(int(math.floor(min(sx[0], sx[1], sx[2]))), 0)
                maxx = min(int(math.ceil(max(sx[0], sx[1], sx[2]))), W - 1)
                miny = max(int(math.floor(min(sy[0], sy[1], sy[2]))), 0)
                maxy = min(int(math.ceil(max(sy[0], sy[1], sy[2]))), H - 1)
                if minx > maxx or miny > maxy:
                    continue
                den = (sy[1] - sy[2]) * (sx[0] - sx[2]) + (sx[2] - sx[1]) * (sy[0] - sy[2])
                if abs(den) < 1e-12:
                    continue
                for py in range(miny, maxy + 1):
                    for px in range(minx, maxx + 1):
                        cx = px + 0.5
                        cy = py + 0.5
                        w0 = ((sy[1] - sy[2]) * (cx - sx[2]) + (sx[2] - sx[1]) * (cy - sy[2])) / den
                        w1 = ((sy[2] - sy[0]) * (cx - sx[2]) + (sx[0] - sx[2]) * (cy - sy[2])) / den
                        w2 = 1.0 - w0 - w1
                        if w0 < -1e-6 or w1 < -1e-6 or w2 < -1e-6:
                            continue
                        z = w0 * sz[0] + w1 * sz[1] + w2 * sz[2]
                        if z < zb[py, px]:
                            zb[py, px] = z
                            for c in range(3):
                                img[py, px, c] = (w0 * col[ai, aj, c] + w1 * col[bi, bj, c] + w2 * col[ci, cj, c])
                            if water[ai, aj] and eye_spec > 0.0:
                                # specular glint on water from each light (Blinn-Phong on a flat surface)
                                wx = x0 + (aj + 0.5) * dx
                                wy = y0 + (ai + 0.5) * dx
                                wz = h[ai, aj]
                                ex, ey, ez = cam[0] - wx, cam[1] - wy, cam[2] - wz
                                en = math.sqrt(ex * ex + ey * ey + ez * ez)
                                for L in range(spec_l.shape[0]):
                                    lxv, lyv, lzv = spec_l[L, 0] - wx, spec_l[L, 1] - wy, spec_l[L, 2] - wz
                                    ln = math.sqrt(lxv * lxv + lyv * lyv + lzv * lzv)
                                    hz = ez / en + lzv / ln
                                    hn = math.sqrt((ex / en + lxv / ln) ** 2 + (ey / en + lyv / ln) ** 2 + hz * hz)
                                    s = (hz / hn) ** 120 * spec_l[L, 3] / (ln * ln) * eye_spec
                                    for c in range(3):
                                        img[py, px, c] += s * spec_rgb[L, c]
    return img, zb


@njit(cache=True)
def _raster_tris(img, zb, tris, cols, cam, fwd, right, up, f):
    H, W = zb.shape
    for t in range(tris.shape[0]):
        sx = np.empty(3)
        sy = np.empty(3)
        sz = np.empty(3)
        ok = True
        for v in range(3):
            wx = tris[t, v, 0] - cam[0]
            wy = tris[t, v, 1] - cam[1]
            wz = tris[t, v, 2] - cam[2]
            zc = wx * fwd[0] + wy * fwd[1] + wz * fwd[2]
            if zc <= 1.0:
                ok = False
                break
            sx[v] = W / 2 + f * (wx * right[0] + wy * right[1] + wz * right[2]) / zc
            sy[v] = H / 2 - f * (wx * up[0] + wy * up[1] + wz * up[2]) / zc
            sz[v] = zc
        if not ok:
            continue
        den = (sy[1] - sy[2]) * (sx[0] - sx[2]) + (sx[2] - sx[1]) * (sy[0] - sy[2])
        if abs(den) < 1e-12:
            continue
        minx = max(int(math.floor(min(sx[0], sx[1], sx[2]))), 0)
        maxx = min(int(math.ceil(max(sx[0], sx[1], sx[2]))), W - 1)
        miny = max(int(math.floor(min(sy[0], sy[1], sy[2]))), 0)
        maxy = min(int(math.ceil(max(sy[0], sy[1], sy[2]))), H - 1)
        for py in range(miny, maxy + 1):
            for px in range(minx, maxx + 1):
                cx = px + 0.5
                cy = py + 0.5
                w0 = ((sy[1] - sy[2]) * (cx - sx[2]) + (sx[2] - sx[1]) * (cy - sy[2])) / den
                w1 = ((sy[2] - sy[0]) * (cx - sx[2]) + (sx[0] - sx[2]) * (cy - sy[2])) / den
                w2 = 1.0 - w0 - w1
                if w0 < -1e-6 or w1 < -1e-6 or w2 < -1e-6:
                    continue
                z = w0 * sz[0] + w1 * sz[1] + w2 * sz[2]
                if z < zb[py, px]:
                    zb[py, px] = z
                    for c in range(3):
                        img[py, px, c] = cols[t, c]


def box(p0, p1, width: float, height: float, z_bottom: float) -> np.ndarray:
    """Triangles of a box from p0 to p1 (plan), ``width`` across, standing from z_bottom up ``height``."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    L = np.hypot(*d)
    u = d / L if L > 0 else np.array([1.0, 0.0])
    v = np.array([-u[1], u[0]]) * width / 2
    corners = [p0 - v, p1 - v, p1 + v, p0 + v]
    b = [np.array([c[0], c[1], z_bottom]) for c in corners]
    t = [np.array([c[0], c[1], z_bottom + height]) for c in corners]
    faces = [(t[0], t[1], t[2]), (t[0], t[2], t[3]), (b[0], b[2], b[1]), (b[0], b[3], b[2])]
    for k in range(4):
        a, c = k, (k + 1) % 4
        faces += [(b[a], b[c], t[c]), (b[a], t[c], t[a])]
    return np.array(faces)


def cylinder(c, r: float, z0: float, z1: float, n: int = 24) -> np.ndarray:
    a = np.linspace(0, 2 * np.pi, n + 1)
    ring = np.column_stack([c[0] + r * np.cos(a), c[1] + r * np.sin(a)])
    faces = []
    for k in range(n):
        p, q = ring[k], ring[k + 1]
        faces += [((p[0], p[1], z0), (q[0], q[1], z0), (q[0], q[1], z1)),
                  ((p[0], p[1], z0), (q[0], q[1], z1), (p[0], p[1], z1)),
                  ((c[0], c[1], z1), (p[0], p[1], z1), (q[0], q[1], z1))]
    return np.array(faces, float)


def scara_mesh(garden: Garden, arm, q, scene: Scene) -> list[tuple[np.ndarray, tuple]]:
    """Walnut links with brass joint caps for a SCARA at joint values q (tray frame + table offsets)."""
    S = scene.base + garden.tray.bed_depth
    lh = S + arm.link_height
    base = arm.base
    elbow = arm.elbows(q)[0]
    tool = arm.fk(q)[0]
    walnut, brass = (0.14, 0.08, 0.045), (0.45, 0.33, 0.14)
    z_tool = S + tool[2]
    parts = [
        (cylinder(base, 42, scene.base + garden.tray.bed_depth, lh + 34), walnut),
        (box(base, elbow, 52, 30, lh), walnut),
        (box(elbow, tool[:2], 42, 24, lh + 32), walnut),
        (cylinder(base, 20, lh + 30, lh + 38), brass),
        (cylinder(elbow, 22, lh + 30, lh + 60), brass),
        (cylinder(tool[:2], 11, z_tool + 18, lh + 32), brass),
    ]
    head_dir = np.array([math.cos(tool[3]), math.sin(tool[3])])
    across = np.array([-head_dir[1], head_dir[0]])
    half = garden.rake.bar_half
    parts.append((box(tool[:2] - across * half, tool[:2] + across * half, garden.rake.bar_thickness, 16,
                      z_tool + 2), walnut))
    return parts


def lantern_mesh(garden: Garden, lan, scene: Scene) -> list:
    """A small stone lantern (kasuga-style, simplified): foot, post, firebox with glowing windows,
    roof, finial. Heights scale with the lantern's configured height."""
    S = scene.base + garden.tray.bed_depth
    H, r, c = lan.height, lan.radius, lan.xy
    stone = (0.30, 0.29, 0.27)
    z = lambda f: S + f * H
    fb0, fb1 = lan.light_height - 0.12 * H, lan.light_height + 0.10 * H
    parts = [
        (cylinder(c, r * 0.95, z(0.0), z(0.12), 6), stone),
        (cylinder(c, r * 0.35, z(0.12), S + fb0 - 0.04 * H, 12), stone),
        (cylinder(c, r * 0.70, S + fb0 - 0.04 * H, S + fb0, 6), stone),
        (cylinder(c, r * 0.55, S + fb0, S + fb1, 6), stone),
        (cylinder(c, r * 1.05, S + fb1, z(0.86), 6), stone),
        (cylinder(c, r * 0.18, z(0.86), z(1.0), 8), stone),
    ]
    # windows: slightly proud of the firebox so they win the depth test, emissive
    glow = cct_to_rgb(lan.cct, white=scene.white)
    parts.append((cylinder(c, r * 0.56, S + fb0 + 0.02 * H, S + fb1 - 0.02 * H, 6), ("emit", glow * 1.4)))
    return parts


def shade_mesh(parts, scene: Scene, ambient_lux: float, ambient_cct: float, key: float, reference: float):
    tris, cols = [], []
    amb_rgb = cct_to_rgb(ambient_cct, white=scene.white)
    for faces, albedo in parts:
        if isinstance(albedo, tuple) and len(albedo) == 2 and albedo[0] == "emit":
            for tri in faces:
                tris.append(tri)
                cols.append(np.minimum(albedo[1] * key, 1.0))
            continue
        for tri in faces:
            nrm = np.cross(tri[1] - tri[0], tri[2] - tri[0])
            nn = np.linalg.norm(nrm)
            if nn == 0:
                continue
            nrm = nrm / nn
            c = tri.mean(axis=0)
            E = ambient_lux * (0.5 + 0.5 * abs(nrm[2])) * amb_rgb
            for (lx, ly, lz, cd, rgb, skip) in scene.lights:
                if math.hypot(c[0] - lx, c[1] - ly) < skip:       # a lantern does not light its own stone
                    continue
                l = np.array([lx, ly, lz]) - c
                d2 = float(l @ l)
                ndl = abs(float(nrm @ l)) / math.sqrt(d2)
                E = E + cd * ndl / d2 * 1e6 * rgb
            rad = np.asarray(albedo) * E / math.pi
            x = rad * (key / reference)
            tris.append(tri)
            cols.append(x / (1 + 0.25 * x))
    return np.array(tris, float), np.array(cols, float)


def perspective(baked: Baked, scene: Scene, eye, target, fov_deg: float, size=(1600, 1000),
                reference: float = 1.0, key: float = 0.5, background=(0.03, 0.03, 0.035),
                meshes=None, ambient_lux: float = 15.0, ambient_cct: float = 3000.0) -> np.ndarray:
    W, H = size
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    f = (W / 2) / math.tan(math.radians(fov_deg) / 2)
    col = expose(baked.radiance, key, reference)
    spec_l = np.array([[l[0], l[1], l[2], l[3] * 1e6] for l in scene.lights]) if scene.lights else np.zeros((0, 4))
    spec_rgb = np.array([l[4] for l in scene.lights]) if scene.lights else np.zeros((0, 3))
    img, zb = _raster(scene.h, col, scene.x0, scene.y0, scene.dx, eye, fwd, right, up, f, W, H,
                      scene.water, spec_l, spec_rgb * (key / reference) * 0.6, 1.0)
    if meshes:
        tris, cols = shade_mesh(meshes, scene, ambient_lux, ambient_cct, key, reference)
        _raster_tris(img, zb, tris, cols, eye, fwd, right, up, f)
    img[np.isinf(zb)] = background
    return to_srgb8(img)
