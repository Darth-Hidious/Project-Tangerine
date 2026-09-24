"""Command line for the twin: plan a pattern into joint-space G-code, and check a G-code file
before it goes anywhere near the real arm.

    karesansui plan ripples -o ripples.gcode              plan for the configured arm, write G-code
    karesansui check ripples.gcode --png ripples.png      strict parse, joint limits, clearances,
                                                          replay on the sand model, top view

Both default to configs/poc.toml and print a JSON summary. Exit status: 0 ok; 1 the program or
the file failed a check (plan then writes nothing); 2 bad arguments or a config without an arm.

What check checks
  - every line is in the subset the interpreter models (anything else is rejected, not skipped);
  - every move ends inside the joint limits (the limits are boxes, so the joint-linear moves
    between those ends stay inside them too);
  - every sampled tool pose either keeps the whole head footprint on the free sand (clear of the
    sand's edge and the stones) or is at travel height, i.e. no more than 5 mm below the travel
    lift, the same margin the planner's clearance check uses for everything within reach.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import shapely

from . import gcode
from .config import POC_CONFIG, load_garden
from .geometry import poses_valid, sand_region
from .patterns import PATTERNS, PatternError
from .planner import ArmMachine, PlanningError, build_program, make_machine
from .sim import Bed, SimCfg

TRAVEL_MARGIN = 5.0     # mm below the travel lift that still counts as travelling


def _plan(args) -> tuple[int, dict]:
    garden = load_garden(args.config)
    if garden.arm is None and (args.output or args.arm):
        return 2, {"error": f"{args.config} has no [arm]; G-code is generated for arm machines only"}
    machine = make_machine(garden, args.arm)
    try:
        prog = build_program(garden, args.pattern, erase=not args.no_erase, machine=machine)
    except (PatternError, PlanningError) as e:
        return 1, {"pattern": args.pattern, "ok": False, "errors": [str(e)]}
    r = prog.report
    out = {"pattern": args.pattern, "machine": machine.name, "ok": r.ok,
           "rake_passes": r.n_rake, "screed_passes": r.n_screed, "rake_m": round(r.rake_mm / 1000, 3),
           "coverage": round(r.coverage, 3), "min_radius_mm": round(r.min_radius_mm, 1),
           "poses_checked": r.poses_checked, "collisions": r.collisions,
           "warnings": r.warnings, "errors": r.errors}
    if not r.ok:
        return 1, out
    if args.output:
        lines = gcode.generate(prog)
        Path(args.output).write_text("\n".join(lines) + "\n")
        out["gcode"] = {"file": str(args.output), "lines": len(lines)}
    return 0, out


def _check(args) -> tuple[int, dict]:
    garden = load_garden(args.config)
    if garden.arm is None:
        return 2, {"error": f"{args.config} has no [arm]"}
    machine = ArmMachine(garden, args.arm)
    arm = machine.arm
    out: dict = {"file": str(args.file), "machine": machine.name}
    try:
        segs = gcode.parse(Path(args.file).read_text().splitlines(), machine.name)
    except ValueError as e:
        return 1, {**out, "ok": False, "errors": [f"rejected: {e}"]}
    if not segs:
        return 1, {**out, "ok": False, "errors": ["no motion in the file"]}
    errors = []
    ends = np.vstack([segs[0].q0] + [s.q1 for s in segs])
    beyond = int((~arm.in_limits(ends)).sum())
    if beyond:
        errors.append(f"{beyond} move end points outside the joint limits")

    cfg = SimCfg()
    traj = gcode.sample(segs, arm, step_mm=cfg.dx * cfg.step)
    poses = np.column_stack([traj["x"], traj["y"], traj["heading"]])
    travelling = traj["z"] >= machine.z_travel - TRAVEL_MARGIN
    unsafe = int((~travelling & ~poses_valid(poses, garden, machine.region)).sum())
    if unsafe:
        errors.append(f"{unsafe} tool samples put the head below travel height off the free sand "
                      "(it would hit an edge, a stone or something else in the garden)")

    bed = Bed(garden, cfg)
    v0 = bed.volume()
    gcode.run_on_bed(bed, traj)
    rel = bed.h[bed.bed] - garden.tray.bed_depth
    out.update({
        "moves": len(segs),
        "machine_minutes": round(traj["seconds"] / 60, 2),
        "joint_limit_violations": beyond,
        "unsafe_samples": unsafe,
        "sim": {"volume_error_rel": float((bed.volume() - v0) / v0),
                "relief_mm": {"min": round(float(rel.min()), 2), "max": round(float(rel.max()), 2),
                              "std": round(float(rel.std()), 3)}},
        "ok": not errors, "errors": errors,
    })
    if args.png:
        _top_view(garden, bed, args.png)
        out["png"] = str(args.png)
    return (0 if not errors else 1), out


def _top_view(garden, bed, path: Path) -> None:
    """Lantern-lit top view of the replayed sand, exposed so the median sand brightness is mid-grey."""
    from matplotlib.image import imsave

    from . import render

    sc = render.build_scene(garden, bed.h, dx=bed.cfg.dx)
    bk = render.bake(sc, ambient_lux=garden.lighting.ambient_lux)
    X, Y = np.meshgrid(sc.x0 + (np.arange(sc.h.shape[1]) + 0.5) * sc.dx,
                       sc.y0 + (np.arange(sc.h.shape[0]) + 0.5) * sc.dx)
    sand = shapely.contains_xy(sand_region(garden), X, Y)
    ref = float(np.median(bk.radiance[sand].mean(axis=1)))
    imsave(path, render.topdown(bk, sc, reference=ref))


def main(argv: list[str] | None = None) -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", type=Path, default=POC_CONFIG,
                        help="garden config (TOML); default: the proof of concept, configs/poc.toml")
    common.add_argument("--arm", choices=["scara", "articulated"], help="arm model (default: the config's)")
    ap = argparse.ArgumentParser(prog="karesansui", description="In-silico twin of the raked table garden.")
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan", parents=[common], help="plan a pattern and, with -o, write joint-space G-code")
    p.add_argument("pattern", choices=sorted(PATTERNS))
    p.add_argument("--no-erase", action="store_true", help="skip the screed passes that flatten the sand first")
    p.add_argument("-o", "--output", type=Path, help="G-code file to write (only if every check passes)")
    c = sub.add_parser("check", parents=[common],
                       help="parse a G-code file strictly, check limits and clearances, replay it on the sand")
    c.add_argument("file", type=Path)
    c.add_argument("--png", type=Path, help="write a lantern-lit top view of the replayed sand")
    args = ap.parse_args(argv)
    code, out = (_plan if args.command == "plan" else _check)(args)
    print(json.dumps(out, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
