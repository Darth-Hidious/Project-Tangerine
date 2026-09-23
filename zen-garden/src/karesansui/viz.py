"""Plan-view drawings and chart styling shared by the experiments.

Colours follow a validated categorical palette (slot 1 blue, slot 2 orange) with ink tokens
for text; see docs for the palette check.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon

from shapely.geometry import LineString, Point, Polygon

from .geometry import sand_region, stone_polygons, tine_paths
from .planner import Pass, Program, Travel, coverage

SURFACE, INK, INK2, MUTED, HAIR, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#d03b3b"
STONE = "#3d3c39"


def style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8, length=3, width=0.8)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)


SAND, WATER, MOSS_LIVE, MOSS_DRY, WOOD, ROCK = "#efe9dc", "#86b6ef", "#3f8f3a", "#a7b98a", "#8a5a3b", "#9a9890"


def _poly(ax, geom, **kw):
    geoms = [geom] if geom.geom_type == "Polygon" else list(getattr(geom, "geoms", []))
    for g in geoms:
        if g.is_empty:
            continue
        ax.add_patch(MplPolygon(np.asarray(g.exterior.coords), closed=True, **kw))


def layout_zones(garden) -> dict:
    """Plan-view zones of an arm garden: sand, water, living and preserved moss."""
    from shapely.geometry import box
    from shapely.ops import unary_union
    tray = box(0, 0, garden.tray.width, garden.tray.depth)
    sand = sand_region(garden)
    water = []
    for f in garden.features:
        if f.kind == "stream":
            water.append(LineString(f.outline).buffer(f.width / 2, cap_style="round"))
        elif f.kind == "pool":
            water.append(Polygon(f.outline))
    water = unary_union(water) if water else Polygon()
    land = tray.difference(sand).difference(water)
    living = water.buffer(45.0).intersection(land) if not water.is_empty else Polygon()
    preserved = land.difference(living)
    return {"tray": tray, "sand": sand, "water": water, "living": living, "preserved": preserved}


def draw_tray(ax, garden, stones: bool = True, layout: bool = True) -> None:
    W, D = garden.tray.width, garden.tray.depth
    if layout and garden.sand_outline:
        z = layout_zones(garden)
        _poly(ax, z["preserved"], color=MOSS_DRY, lw=0, zorder=0)
        _poly(ax, z["living"], color=MOSS_LIVE, lw=0, zorder=0)
        _poly(ax, z["water"], color=WATER, lw=0, zorder=0.5)
        _poly(ax, z["sand"], color=SAND, lw=0, zorder=0.2)
        for f in garden.features:
            if f.kind == "rock":
                _poly(ax, Polygon(f.outline), color=ROCK, lw=0, zorder=1)
            elif f.kind == "bridge":
                _poly(ax, Polygon(f.outline), color=WOOD, lw=0, zorder=1)
        for lan in garden.lanterns:
            ax.add_patch(matplotlib.patches.Circle(lan.xy, lan.radius, color="#6f6e69", lw=0, zorder=6))
            ax.add_patch(matplotlib.patches.Circle(lan.xy, lan.radius * 0.45, color="#f3c77a", lw=0, zorder=7))
        if garden.arm is not None:
            ax.add_patch(matplotlib.patches.Circle(garden.arm.base, garden.arm.base_radius, color=WOOD, lw=0, zorder=6))
    ax.plot([0, W, W, 0, 0], [0, 0, D, D, 0], color=INK2, lw=1.0, zorder=1)
    if stones:
        for poly in stone_polygons(garden):
            ax.add_patch(MplPolygon(np.asarray(poly.exterior.coords), closed=True, color=STONE, zorder=5, lw=0))
    ax.set_xlim(-15, W + 15)
    ax.set_ylim(-15, D + 15)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)


def plot_plan(ax, program: Program, show_travel: bool = True, show_gaps: bool = True) -> None:
    g = program.garden
    draw_tray(ax, g)
    rakes = [p for p in program.passes if p.kind == "rake"]
    if show_gaps:
        _, raked = coverage(rakes, g)
        cell = 2.0
        xs, ys = np.arange(cell / 2, g.tray.width, cell), np.arange(cell / 2, g.tray.depth, cell)
        X, Y = np.meshgrid(xs, ys)
        import shapely
        grit = shapely.contains_xy(sand_region(g), X, Y)
        gaps = np.ma.masked_where(raked | ~grit, np.ones(raked.shape))
        ax.imshow(gaps, extent=(0, g.tray.width, 0, g.tray.depth), origin="lower", cmap=matplotlib.colors.ListedColormap([ORANGE]),
                  alpha=0.16, zorder=0, interpolation="nearest")
    lw = 0.7 if g.rake.pitch >= 20 else 0.45
    for p in rakes:
        for path in tine_paths(p.poses[:, :2], g.rake.tine_offsets):
            ax.plot(path[:, 0], path[:, 1], color=BLUE, lw=lw, solid_capstyle="round", zorder=3)
    if show_travel:
        for s in program.steps:
            if isinstance(s, Travel):
                ax.plot(s.poses[:, 0], s.poses[:, 1], color=MUTED, lw=0.5, alpha=0.8, zorder=2)


def save(fig, path) -> None:
    fig.savefig(path, facecolor=SURFACE, dpi=fig.dpi)
    plt.close(fig)


def draw_scara(ax, arm, q, color=WOOD, alpha=1.0) -> None:
    """Top view of a SCARA configuration: base, two links, the lift column."""
    q = np.atleast_2d(q)[0]
    e = arm.elbows(q)[0]
    t = arm.fk(q)[0, :2]
    ax.plot([arm.base[0], e[0], t[0]], [arm.base[1], e[1], t[1]], color=color, lw=7, alpha=alpha,
            solid_capstyle="round", zorder=8)
    for pt in (arm.base, e, t):
        ax.add_patch(matplotlib.patches.Circle(pt, 9, color="#b08d57", lw=0, zorder=9, alpha=alpha))
