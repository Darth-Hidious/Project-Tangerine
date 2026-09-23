"""Which arm, which drives? SCARA vs articulated over the proof-of-concept sand bed.

For every rake pose of every pattern the planner produces, the script computes the joint
configuration, the holding torques and, for four real drive types, the worst-case wobble
of the grooves (tool-point error across the direction of travel). It also maps the
worst-case wobble over the whole sand bed (worst heading at each point).

Drive data (joint output, degrees):
  Feetech STS3215 hobby servo    backlash 0.50 (datasheet max; a bench test measured 0.77),
                                 resolution 0.088 (12-bit encoder)
  Dynamixel XM430-W350           backlash 0.25 (15 arcmin, ROBOTIS spec), resolution 0.088
  NEMA17 stepper + 5:1 GT2 belt  step accuracy +-5 % of 1.8 deg at the motor -> +-0.018 at the
                                 joint, taken as a 0.036 dead band; 1/16 microstep -> 0.0225;
                                 belt backlash taken as zero (tensioned GT2) - an estimate
  Harmonic drive + output encoder  lost motion < 1 arcmin (0.017); 14-bit encoder 0.022

Run:  python experiments/arm_study.py      (writes docs/figures/arm_wobble.png, docs/arm_study.json)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shapely

from karesansui.arm import Drive, gravity_preloaded, line_wobble
from karesansui.config import poc_garden
from karesansui.geometry import sand_region
from karesansui.planner import ArmMachine, build_program
from karesansui import viz

ROOT = Path(__file__).resolve().parents[1]
DRIVES = [
    Drive("Hobby servo (STS3215)", backlash=0.50, resolution=0.088),
    Drive("Smart servo (Dynamixel XM430)", backlash=0.25, resolution=0.088),
    Drive("Stepper + 5:1 belt", backlash=0.036, resolution=0.0225),
    Drive("Harmonic drive", backlash=0.017, resolution=0.022),
]
DRAG_N = 0.5          # rake drag: skid friction (~60 g head, mu ~0.5) plus nine 2.5 mm tines; conservative


def rake_poses(garden, machine):
    poses = []
    for pattern in ("lines", "waves", "ripples", "spiral"):
        prog = build_program(garden, pattern, machine=machine)
        assert prog.report.ok, prog.report.errors
        poses += [p.poses for p in prog.passes if p.kind == "rake"]
    return np.vstack(poses)


def main() -> dict:
    garden = poc_garden()
    out = {}
    maps = {}
    region = sand_region(garden).buffer(-(garden.planner.wall_clearance + garden.rake.tine_half))
    xs, ys = np.arange(200, 560, 4.0), np.arange(40, 420, 4.0)
    X, Y = np.meshgrid(xs, ys)
    inside = shapely.contains_xy(region, X, Y)
    for kind in ("scara", "articulated"):
        m = ArmMachine(garden, kind)
        arm = m.arm
        poses = rake_poses(garden, m)
        q, ok = m.ik(poses, m.z_for("rake"))
        assert ok.all()
        pre = gravity_preloaded(arm, q)
        tau = np.abs(arm.gravity_torques(q))
        res = {
            "joints": list(arm.joint_names),
            "joint_range_deg": [[round(float(np.degrees(q[:, k].min())), 1), round(float(np.degrees(q[:, k].max())), 1)]
                                if k != (2 if kind == "scara" else -1) else
                                [round(float(q[:, k].min()), 1), round(float(q[:, k].max()), 1)]
                                for k in range(arm.n_joints)],
            "gravity_preloaded": [bool(v) for v in pre],
            "max_holding_torque_Nm": [round(float(v), 3) for v in tau.max(axis=0)],
            "wobble_mm": {},
        }
        if kind == "scara":
            inertia = arm.inertia_about_base(q)
            r = np.hypot(*(poses[:, :2] - arm.base).T) / 1000
            alpha = garden.gantry.accel / 1000 / r
            res["base_torque_accel_plus_drag_Nm"] = round(float((inertia * alpha + DRAG_N * r).max()), 3)
        for drive in DRIVES:
            centre, smear = line_wobble(arm, q, poses[:, 2], drive, garden.rake.tine_offsets, pre)
            res["wobble_mm"][drive.name] = {"p50": round(float(np.percentile(centre, 50)), 3),
                                            "p95": round(float(np.percentile(centre, 95)), 3),
                                            "max": round(float(centre.max()), 3)}
        # Map: worst heading at every point of the reachable sand.
        pts = np.column_stack([X[inside], Y[inside]])
        worst = {d.name: np.zeros(len(pts)) for d in DRIVES}
        for h in np.radians(np.arange(0, 180, 10)):
            pp = np.column_stack([pts, np.full(len(pts), h)])
            qg, okg = m.ik(pp, m.z_for("rake"))
            for d in DRIVES:
                c, _ = line_wobble(arm, qg, pp[:, 2], d, garden.rake.tine_offsets, pre)
                worst[d.name] = np.maximum(worst[d.name], np.where(okg, c, np.nan))
        maps[kind] = worst
        out[kind] = res
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "arm_study.json").write_text(json.dumps(out, indent=2))
    plot(garden, maps, X, Y, inside)
    return out


def plot(garden, maps, X, Y, inside) -> None:
    kinds = list(maps)
    fig, axes = plt.subplots(len(kinds), len(DRIVES), figsize=(13.2, 6.5), dpi=150, facecolor=viz.SURFACE)
    # One sequential blue ramp for magnitude, shared scale; above 2 mm is off the scale (clipped).
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    norm = matplotlib.colors.Normalize(0, 2.0)
    for r, kind in enumerate(kinds):
        for c, drive in enumerate(DRIVES):
            ax = axes[r, c]
            viz.draw_tray(ax, garden)
            grid = np.full(X.shape, np.nan)
            grid[inside] = maps[kind][drive.name]
            ax.imshow(np.clip(grid, 0, 2.0), extent=(X.min() - 2, X.max() + 2, Y.min() - 2, Y.max() + 2),
                      origin="lower", cmap=cmap, norm=norm, zorder=2, interpolation="nearest")
            ax.set_xlim(190, 700)
            ax.set_ylim(20, 440)
            p95 = np.nanpercentile(maps[kind][drive.name], 95)
            if r == 0:
                ax.set_title(drive.name, fontsize=9.5, color=viz.INK, loc="left")
            label = "SCARA" if kind == "scara" else "Articulated"
            ax.set_xlabel(f"{label}: 95% of the bed ≤ {p95:.2f} mm", fontsize=8.5, color=viz.INK2, labelpad=3)
    cax = fig.add_axes([0.35, 0.075, 0.3, 0.022])
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
    cb.set_label("worst-case groove wobble across the line (mm), worst heading; clipped at 2 mm",
                 fontsize=8.5, color=viz.INK2)
    cb.ax.tick_params(labelsize=8, colors=viz.MUTED)
    cb.outline.set_visible(False)
    fig.suptitle("Where the rake's lines would wobble, by arm type and joint drive (8 mm groove pitch)",
                 x=0.012, ha="left", fontsize=12, color=viz.INK, fontweight="bold")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.19, wspace=0.04, hspace=0.22)
    (ROOT / "docs" / "figures").mkdir(parents=True, exist_ok=True)
    viz.save(fig, ROOT / "docs" / "figures" / "arm_wobble.png")


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
