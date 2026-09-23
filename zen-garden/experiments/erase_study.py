"""Does the sand stay where it belongs over many erase-and-rake cycles?

The blade drops whatever it still carries where a pass ends. Where that lands on sand no pass
sweeps (the strip along the edge, the gap between the stones), it stays, and the next cycle adds
more. This runs the proof-of-concept garden through N cycles (ripples and lines in turn, through
G-code, on one bed) twice: with the erase the planner uses, and with the erase as first written
(lanes and rings, the blade lifted at once). It tracks the sand standing more than 2 mm above
the flat bed and the 99.9th-percentile height.

Run:  python experiments/erase_study.py [cycles]     (40 cycles: about two minutes)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from karesansui import gcode, viz
from karesansui.config import poc_garden
from karesansui.planner import ArmMachine, build_program
from karesansui.sim import Bed, SimCfg

ROOT = Path(__file__).resolve().parents[1]
PILE = 2.0          # mm above the flat bed that counts as piled up


def run(garden, machine, cycles: int, bare: bool) -> list[dict]:
    cfg = SimCfg(dx=1.0)
    progs = [build_program(garden, pat, machine=machine, bare_erase=bare) for pat in ("ripples", "lines")]
    trajs = [gcode.sample(gcode.parse(gcode.generate(p), machine.name), machine.arm, step_mm=cfg.dx * cfg.step)
             for p in progs]
    bed = Bed(garden, cfg)
    v0 = bed.volume()
    rows = []
    for k in range(cycles):
        gcode.run_on_bed(bed, trajs[k % 2])
        rel = bed.h[bed.bed] - garden.tray.bed_depth
        rows.append({"cycle": k + 1,
                     "piled_mm3": round(float(np.clip(rel - PILE, 0, None).sum()) * cfg.dx**2, 2),
                     "p999_mm": round(float(np.percentile(rel, 99.9)), 3),
                     "max_mm": round(float(rel.max()), 3),
                     "std_mm": round(float(rel.std()), 4),
                     "volume_error_rel": float((bed.volume() - v0) / v0)})
    return rows


def plot(res: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150, facecolor=viz.SURFACE)
    series = [("full", "erase the planner uses", viz.BLUE), ("bare", "erase as first written", viz.ORANGE)]
    panels = [("piled_mm3", f"sand more than {PILE:g} mm above the flat bed (mm³)"),
              ("p999_mm", "99.9th-percentile height above the flat bed (mm)")]
    for ax, (key, label) in zip(axes, panels):
        viz.style_axes(ax)
        ax.grid(axis="y", color=viz.HAIR, lw=0.6)
        ax.set_axisbelow(True)
        for name, text, colour in series:
            x = [r["cycle"] for r in res[name]]
            y = [r[key] for r in res[name]]
            ax.plot(x, y, color=colour, lw=2, label=text)
            ax.annotate(f"{y[-1]:g}", (x[-1], y[-1]), xytext=(4, 0), textcoords="offset points",
                        fontsize=8.5, color=viz.INK2, va="center")
        ax.set_xlabel("erase-and-rake cycles")
        ax.set_title(label, loc="left", fontsize=9.5, color=viz.INK)
        ax.set_xlim(0, len(res["full"]) + 5)
    axes[0].legend(frameon=False, fontsize=8.5, labelcolor=viz.INK2, loc="upper left")
    fig.suptitle("Repeated cycles on one bed: the planner's erase levels off, the first version piles sand up",
                 x=0.01, ha="left", fontsize=11.5, color=viz.INK, fontweight="bold")
    fig.tight_layout()
    viz.save(fig, ROOT / "docs" / "figures" / "erase_cycles.png")


def main(cycles: int = 40) -> dict:
    garden = poc_garden()
    machine = ArmMachine(garden, "scara")
    res = {"full": run(garden, machine, cycles, bare=False), "bare": run(garden, machine, cycles, bare=True)}
    (ROOT / "docs" / "erase_study.json").write_text(json.dumps(res, indent=1))
    plot(res)
    return {k: v[-1] for k, v in res.items()}


if __name__ == "__main__":
    print(json.dumps(main(int(sys.argv[1]) if len(sys.argv) > 1 else 40), indent=2))
