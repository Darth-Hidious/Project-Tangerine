"""The proof of concept, end to end, in silico.

For each pattern: plan -> joint-space G-code -> interpreter -> sand simulation -> lantern-lit
render. Also the layout drawing, a groove cross-section, the light on the sand and the water
budget. Every number that PLAN.md quotes is written to docs/poc_results.json by this script.

Run:  python experiments/poc_run.py            (about a minute on a laptop)
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shapely
from PIL import Image
from shapely.geometry import Point

from karesansui import gcode, render, viz, water
from karesansui.config import poc_garden
from karesansui.geometry import sand_region
from karesansui.planner import ArmMachine, build_program
from karesansui.sim import Bed, SimCfg

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"
PATTERNS = ["lines", "waves", "ripples", "spiral"]


def layout_figure(garden, machine) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 7.4), dpi=150, facecolor=viz.SURFACE)
    viz.draw_tray(ax, garden)
    zones = viz.layout_zones(garden)
    reach = matplotlib.patches.Circle(garden.arm.base, machine.arm.l1 + machine.arm.l2, fill=False,
                                      ec=viz.INK2, lw=0.8, alpha=0.6, zorder=2)
    ax.add_patch(reach)
    reach.set_clip_path(matplotlib.patches.Rectangle((0, 0), garden.tray.width, garden.tray.depth,
                                                     transform=ax.transData))
    for f in garden.features:
        if f.kind in ("bay", "reservoir"):
            xy = np.asarray(f.outline + (f.outline[0],))
            ax.plot(xy[:, 0], xy[:, 1], color=viz.INK2, lw=0.9, zorder=3)
    q, _ = machine.ik(np.array([[330.0, 330.0, 0.0]]), machine.z_for("rake"))
    viz.draw_scara(ax, machine.arm, q, alpha=0.9)
    notes = [
        ((115, 250), "stream\n(pumped, ~1 L/min)"),
        ((110, 62), "pool"),
        ((380, 330), "fine white sand\n9 dm², raked"),
        ((398, 212), "stone pair"),
        ((205, 440), "lantern"),
        ((560, 20), "lantern"),
        ((640, 158), "arm base"),
        ((638, 400), "electronics drawer\n(under the dry side)"),
        ((120, 150), "reservoir 2 L\n(under the pool)"),
        ((142, 192), "living\nmoss"),
        ((52, 400), "bonsai"),
        ((470, 22), "preserved moss"),
    ]
    for (x, y), text in notes:
        ax.text(x, y, text, fontsize=8.2, color=viz.INK, ha="center", va="center", zorder=12,
                bbox=dict(boxstyle="round,pad=0.25", fc=viz.SURFACE, ec="none", alpha=0.85))
    ax.text(0, -12, f"SCARA reach {machine.arm.l1 + machine.arm.l2:.0f} mm (circle). "
                    "Tray 700 × 450 mm, viewer at the bottom edge.",
            fontsize=8.5, color=viz.INK2, va="top")
    ax.set_ylim(-35, 465)
    ax.set_title("Proof-of-concept layout (plan view)", loc="left", fontsize=12, color=viz.INK, fontweight="bold")
    fig.tight_layout()
    viz.save(fig, FIG / "poc_layout.png")


def patterns_figure(garden, programs) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.6), dpi=150, facecolor=viz.SURFACE)
    for ax, pat in zip(axes.ravel(), PATTERNS):
        prog = programs[pat]
        r = prog.report
        viz.plot_plan(ax, prog, show_travel=False)
        ax.set_xlim(200, 560)
        ax.set_ylim(30, 425)
        ax.set_title(f"{pat}: {r.n_rake} rake passes, {r.rake_mm / 1000:.2f} m, raked {r.coverage:.0%} of the sand",
                     loc="left", fontsize=9.5, color=viz.INK)
    fig.suptitle("Planned grooves for each pattern (blue); unraked sand tinted orange", x=0.01, ha="left",
                 fontsize=12, color=viz.INK, fontweight="bold")
    fig.tight_layout()
    viz.save(fig, FIG / "poc_patterns.png")


def simulate(garden, machine, prog, cfg):
    lines = gcode.generate(prog)
    segs = gcode.parse(lines, machine.name)
    traj = gcode.sample(segs, machine.arm, step_mm=cfg.dx * cfg.step)
    bed = Bed(garden, cfg)
    v0 = bed.volume()
    t = time.time()
    gcode.run_on_bed(bed, traj)
    wall = time.time() - t
    rel = bed.h[bed.bed] - garden.tray.bed_depth
    stats = {
        "gcode_lines": len(lines),
        "machine_minutes": round(traj["seconds"] / 60, 2),
        "sim_seconds": round(wall, 1),
        "volume_error_rel": float((bed.volume() - v0) / v0),
        "lost_mm3": float(bed.lost_mm3),
        "max_slope_excess_mm": round(bed.max_slope_excess(), 4),
        "relief_mm": {"min": round(float(rel.min()), 2), "max": round(float(rel.max()), 2),
                      "std": round(float(rel.std()), 3)},
    }
    return bed, lines, stats


def render_views(garden, machine, prog, bed, name, dx=1.0, size=(1400, 1000)):
    sc = render.build_scene(garden, bed.h, dx=dx)
    bk = render.bake(sc, ambient_lux=garden.lighting.ambient_lux)
    X, Y = np.meshgrid(sc.x0 + (np.arange(sc.h.shape[1]) + 0.5) * sc.dx, sc.y0 + (np.arange(sc.h.shape[0]) + 0.5) * sc.dx)
    sand = shapely.contains_xy(sand_region(garden), X, Y)
    ref = float(np.median(bk.radiance[sand].mean(axis=1)))
    rk = [p for p in prog.passes if p.kind == "rake"][-1]
    pose = rk.poses[len(rk.poses) // 3]
    q, _ = machine.ik(pose[None], machine.z_for("rake"))
    parts = render.scara_mesh(garden, machine.arm, q[0], sc)
    for lan in garden.lanterns:
        parts += render.lantern_mesh(garden, lan, sc)
    for f in garden.features:
        if f.kind == "tree":
            parts += render.tree_mesh(garden, f, sc)
    img = render.perspective(bk, sc, eye=(330, -560, 620), target=(390, 230, 70), fov_deg=46, size=size,
                             reference=ref, meshes=parts, ambient_lux=garden.lighting.ambient_lux)
    # the hero goes in the docs; per-pattern views only feed the contact sheet (docs/figures/poc_renders.png)
    Image.fromarray(img).save((FIG if name == "hero" else ROOT / "out") / f"poc_render_{name}.png")
    lux = bk.lux[sand]
    return img, sc, bk, sand, {"p5": round(float(np.percentile(lux, 5)), 1),
                               "median": round(float(np.median(lux)), 1),
                               "p95": round(float(np.percentile(lux, 95)), 1)}


def groove_figure(garden, bed) -> dict:
    """Cross-section through the ring grooves below the stone pair, perpendicular to them."""
    dx = bed.cfg.dx
    x_mm = 430.0
    j = int(x_mm / dx)
    ys = (np.arange(bed.ny) + 0.5) * dx
    sel = (ys > 70) & (ys < 150) & bed.bed[:, j]
    prof = bed.h[sel, j] - garden.tray.bed_depth
    y = ys[sel]
    fig, ax = plt.subplots(figsize=(9.5, 2.6), dpi=150, facecolor=viz.SURFACE)
    viz.style_axes(ax)
    ax.axhline(0, color=viz.AXIS, lw=0.8)
    ax.plot(y, prof, color=viz.BLUE, lw=2)
    ax.set_xlabel("position across the grooves (mm)")
    ax.set_ylabel("height vs. flat bed (mm)")
    ax.set_aspect(4.0)
    pk = float(prof.max() - prof.min())
    ax.set_title(f"Simulated cross-section at x = {x_mm:.0f} mm: {garden.rake.pitch:g} mm pitch, "
                 f"{pk:.1f} mm crest to trough (height exaggerated 4x)", loc="left", fontsize=10, color=viz.INK)
    fig.tight_layout()
    viz.save(fig, FIG / "groove_profile.png")
    return {"x_mm": x_mm, "peak_to_trough_mm": round(pk, 2)}


def lux_figure(garden, sc, bk, sand) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 6.0), dpi=150, facecolor=viz.SURFACE)
    viz.draw_tray(ax, garden)
    grid = np.where(sand, bk.lux, np.nan)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    norm = matplotlib.colors.LogNorm(10, 300)
    ext = (sc.x0, sc.x0 + sc.h.shape[1] * sc.dx, sc.y0, sc.y0 + sc.h.shape[0] * sc.dx)
    ax.imshow(np.clip(grid, 10, 300), extent=ext, origin="lower", cmap=cmap, norm=norm, zorder=3, interpolation="nearest")
    ax.set_xlim(180, 620)
    ax.set_ylim(20, 440)
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.04, pad=0.02)
    cb.set_ticks([10, 20, 50, 100, 200, 300])
    cb.set_ticklabels(["10", "20", "50", "100", "200", "300"])
    cb.set_label("illuminance on the sand (lux, log scale)", color=viz.INK2, fontsize=9)
    cb.ax.tick_params(labelsize=8, colors=viz.MUTED)
    cb.outline.set_visible(False)
    ax.set_title("Two 60 lm lanterns + dim room light on the raked sand", loc="left", fontsize=11, color=viz.INK)
    fig.tight_layout()
    viz.save(fig, FIG / "poc_lux.png")


def main() -> dict:
    FIG.mkdir(parents=True, exist_ok=True)
    (ROOT / "out").mkdir(exist_ok=True)
    garden = poc_garden()
    machine = ArmMachine(garden, "scara")
    programs = {p: build_program(garden, p, machine=machine) for p in PATTERNS}
    layout_figure(garden, machine)
    patterns_figure(garden, programs)
    cfg = SimCfg(dx=0.5)
    out = {"patterns": {}}
    renders = []
    for pat in PATTERNS:
        prog = programs[pat]
        r = prog.report
        bed, lines, stats = simulate(garden, machine, prog, cfg)
        (ROOT / "out").mkdir(exist_ok=True)
        (ROOT / "out" / f"poc_{pat}.gcode").write_text("\n".join(lines) + "\n")
        img, sc, bk, sand, lux = render_views(garden, machine, prog, bed, pat)
        renders.append(img)
        out["patterns"][pat] = {
            "rake_passes": r.n_rake, "screed_passes": r.n_screed,
            "rake_m": round(r.rake_mm / 1000, 2), "screed_m": round(r.screed_mm / 1000, 2),
            "coverage": round(r.coverage, 3), "erase_coverage": round(r.erase_coverage, 3),
            "min_radius_mm": round(r.min_radius_mm, 1),
            "warnings": r.warnings, **stats, "sand_lux": lux,
        }
        if pat == "ripples":
            out["groove"] = groove_figure(garden, bed)
            lux_figure(garden, sc, bk, sand)
            hero = render_views(garden, machine, prog, bed, "hero", dx=0.5, size=(1600, 1150))
    # 2 x 2 contact sheet of the renders
    h, w = renders[0].shape[:2]
    sheet = np.zeros((2 * h, 2 * w, 3), np.uint8)
    for k, img in enumerate(renders):
        sheet[(k // 2) * h:(k // 2 + 1) * h, (k % 2) * w:(k % 2 + 1) * w] = img
    Image.fromarray(sheet).resize((w, h)).save(FIG / "poc_renders.png")
    wr = water.report(garden)
    out["water"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(wr).items()}
    q, hd = water.operating_point(3.0, 4.0, garden.water.lift_mm / 1000, garden.water.tube_id_mm / 1000,
                                  garden.water.tube_len_m)
    out["water"]["example_pump_unthrottled"] = {"h_max_m": 3.0, "q_max_lpm": 4.0,
                                                "flow_lpm": round(q, 2), "head_m": round(hd, 2)}
    (ROOT / "docs" / "poc_results.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
