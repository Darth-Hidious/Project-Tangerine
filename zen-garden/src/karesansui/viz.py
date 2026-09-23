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

from .geometry import stone_polygons, tine_paths
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


def draw_tray(ax, garden, stones: bool = True) -> None:
    W, D = garden.tray.width, garden.tray.depth
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
        grit = np.ones_like(raked)
        cell = 2.0
        gaps = np.ma.masked_where(raked, np.ones(raked.shape))
        ax.imshow(gaps, extent=(0, g.tray.width, 0, g.tray.depth), origin="lower", cmap=matplotlib.colors.ListedColormap([ORANGE]),
                  alpha=0.16, zorder=0, interpolation="nearest")
    for p in rakes:
        for path in tine_paths(p.poses[:, :2], g.rake.tine_offsets):
            ax.plot(path[:, 0], path[:, 1], color=BLUE, lw=0.7, solid_capstyle="round", zorder=3)
    if show_travel:
        for s in program.steps:
            if isinstance(s, Travel):
                ax.plot(s.poses[:, 0], s.poses[:, 1], color=MUTED, lw=0.5, alpha=0.8, zorder=2)


def save(fig, path) -> None:
    fig.savefig(path, facecolor=SURFACE, dpi=fig.dpi)
    plt.close(fig)
