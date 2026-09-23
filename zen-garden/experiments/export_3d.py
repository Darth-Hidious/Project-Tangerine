"""Export the twin for the 3D viewer and write docs/garden3d.html (viewer/garden3d.template.html
with the data inlined, so the page opens straight from disk).

What goes in, all computed here rather than drawn by hand:
  - the terrain around the sand (landscape.terrain): ground, zones, water surface;
  - for every pattern, the sand after the simulated cycle and the arm's joint trajectory
    (joint-linear between G-code points, as the controller moves) with the tool state;
  - the layout (stones, rocks, bridge, lanterns, tree, stream, cascades), the arm's geometry,
    and the plan-view envelopes the head and the links can sweep inside the joint limits;
  - the water budget and each program's report.

Run:  python experiments/export_3d.py            (about a minute)
      python experiments/export_3d.py --reuse    (re-inline the last data after editing the template)
"""

from __future__ import annotations

import base64
import json
import math
from pathlib import Path

import numpy as np
import shapely
from scipy.ndimage import distance_transform_edt
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from karesansui import gcode, water
from karesansui.config import poc_garden
from karesansui.geometry import sand_region, swing_radius
from karesansui.landscape import terrain
from karesansui.planner import ArmMachine, build_program
from karesansui.sim import Bed, SimCfg

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = ["lines", "waves", "ripples", "spiral"]
TERRAIN_DX = 2.0          # mm per terrain cell
SAND_DX = 1.0             # mm per sand cell in the viewer (the simulation runs at 0.5)
PATH_STEP = 2.0           # mm of tool travel between trajectory samples


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def poly(p, step: float = 0.0) -> list:
    p = p.simplify(step) if step else p
    geoms = [p] if p.geom_type == "Polygon" else list(p.geoms)
    return [[[round(x, 1), round(y, 1)] for x, y in g.exterior.coords[:-1]] for g in geoms]


def trajectory(segs, arm) -> dict:
    """Joint samples roughly every PATH_STEP mm of tool travel, with time and tool state."""
    t, q, rel = [0.0], [segs[0].q0], [not segs[0].latched]
    clock = 0.0
    for s in segs:
        a, b = arm.fk(s.q0)[0], arm.fk(s.q1)[0]
        n = max(int(math.ceil(max(np.hypot(*(b[:2] - a[:2])), abs(b[2] - a[2])) / PATH_STEP)), 1)
        for k in range(1, n + 1):
            f = k / n
            q.append(s.q0 + f * (s.q1 - s.q0))
            t.append(clock + f * s.seconds)
            rel.append(not s.latched)
        clock += s.seconds
    q = np.array(q)
    return {"n": len(t), "seconds": round(clock, 2),
            "t": b64(np.array(t, np.float32)), "q": b64(q.astype(np.float32)),
            "released": b64(np.array(rel, np.uint8))}


def main() -> dict:
    g = poc_garden()
    m = ArmMachine(g, "scara")
    W, D, S = g.tray.width, g.tray.depth, g.tray.bed_depth

    # ---- terrain (2 mm)
    t = terrain(g, dx=TERRAIN_DX, overlays=False)
    wet = ~np.isnan(t.water)
    d, (ii, jj) = distance_transform_edt(~wet, return_indices=True)
    reach = t.water[ii, jj]                          # water surface carried a few cells under the banks
    reach[d > 3] = np.nan
    ground = {"nx": len(t.x), "ny": len(t.y), "dx": TERRAIN_DX,
              "ground": b64(np.round(t.ground * 10).astype(np.int16)),
              "zone": b64(t.zone.astype(np.uint8)),
              "water": b64(np.where(np.isnan(reach), -32768, np.round(reach * 10)).astype(np.int16)),
              "wet": b64(wet.astype(np.uint8))}

    # ---- sand patterns and programs
    sand = sand_region(g)
    x0, y0, x1, y1 = (math.floor(sand.bounds[0]) - 2, math.floor(sand.bounds[1]) - 2,
                      math.ceil(sand.bounds[2]) + 2, math.ceil(sand.bounds[3]) + 2)
    cfg = SimCfg(dx=0.5)
    k = int(round(SAND_DX / cfg.dx))
    patterns = {}
    for pat in PATTERNS:
        prog = build_program(g, pat, machine=m)
        r = prog.report
        segs = gcode.parse(gcode.generate(prog), "scara")
        bed = Bed(g, cfg)
        gcode.run_on_bed(bed, gcode.sample(segs, m.arm, step_mm=cfg.dx * cfg.step))
        rel = bed.h - S
        rel = rel.reshape(rel.shape[0] // k, k, rel.shape[1] // k, k).mean(axis=(1, 3))
        crop = rel[y0:y1, x0:x1]
        patterns[pat] = {
            "sand": b64(np.round(crop * 100).astype(np.int16)),          # 0.01 mm
            "path": trajectory(segs, m.arm),
            "report": {"rake_passes": r.n_rake, "screed_passes": r.n_screed, "rake_m": round(r.rake_mm / 1000, 2),
                       "coverage": round(r.coverage, 3), "erase_coverage": round(r.erase_coverage, 3)},
        }
        print(pat, "samples", patterns[pat]["path"]["n"], "minutes", round(patterns[pat]["path"]["seconds"] / 60, 2))
    mask = shapely.contains_xy(sand, *np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5))

    # ---- what the arm can sweep inside its joint limits (plan view)
    sw = m.sweep()
    head_r = swing_radius(g.rake, g.screed)
    head_env = unary_union(shapely.buffer(sw["head"][::3], head_r, quad_segs=4)).simplify(1.0)
    link_env = unary_union(shapely.buffer(sw["links"][::5], 21.0, quad_segs=2, cap_style="flat")).simplify(1.5)

    # ---- layout
    feats = []
    for f in g.features:
        if f.kind in ("bay", "reservoir"):
            continue
        feats.append({"name": f.name, "kind": f.kind, "outline": [list(p) for p in f.outline],
                      "height": f.height, "width": f.width, "levels": list(f.levels), "plunge": f.plunge})
    stream = next(f for f in g.features if f.kind == "stream")
    reaches, drops = water.stream_profile(stream)
    cascades = []
    for dr in drops:
        i = dr["at"]
        v, u = np.asarray(stream.outline[i]), np.asarray(stream.outline[i - 1])
        into = (v - u) / np.linalg.norm(v - u)
        lip = v - into * stream.plunge
        cascades.append({"lip": lip.round(1).tolist(), "centre": v.tolist(), "dir": into.round(4).tolist(),
                         "top": stream.levels[i - 1] - water.LIP_RUN, "bottom": stream.levels[i],
                         "height": dr["height_mm"]})
    wr = water.report(g)
    arm = g.arm
    data = {
        "tray": {"width": W, "depth": D, "bed": S, "wall": g.tray.wall_height},
        "terrain": ground,
        "sand": {"x0": x0, "y0": y0, "nx": x1 - x0, "ny": y1 - y0, "dx": SAND_DX,
                 "mask": b64(mask.astype(np.uint8)), "outline": [list(p) for p in g.sand_outline]},
        "patterns": patterns,
        "stones": [{"name": s.name, "outline": [list(p) for p in s.outline], "height": s.height - S} for s in g.stones],
        "features": feats,
        "cascades": cascades,
        "lanterns": [{"name": l.name, "xy": list(l.xy), "radius": l.radius, "height": l.height,
                      "light": l.light_height, "lumens": l.lumens, "cct": l.cct} for l in g.lanterns],
        "arm": {"base": list(arm.base), "baseRadius": arm.base_radius, "zero": arm.zero_deg,
                "l1": arm.scara.link1, "l2": arm.scara.link2, "linkHeight": arm.scara.link_height,
                "w1": arm.scara.link1_width, "w2": arm.scara.link2_width, "travel": arm.travel_lift,
                "lim1": list(arm.scara.j1_limits), "lim2": list(arm.scara.j2_limits)},
        "head": {"tines": g.rake.n_tines, "pitch": g.rake.pitch, "tineWidth": g.rake.tine_width,
                 "depth": g.rake.depth, "blade": g.screed.width, "swing": round(head_r, 1),
                 "tineHalf": g.rake.tine_half},
        "envelope": {"head": poly(head_env), "links": poly(link_env)},
        "water": {"stream_mm": round(wr.stream_length_m * 1000), "open_dm2": round(wr.open_water_m2 * 100, 2),
                  "evap_l_day": [round(v, 2) for v in wr.evaporation_l_day],
                  "refill_days": [round(v, 1) for v in wr.autonomy_days],
                  "reservoir_l": g.water.reservoir_l, "flow_lpm": g.water.flow_lpm,
                  "cascades": [round(c["height_mm"]) for c in wr.cascades]},
        "sandArea_dm2": round(sand.area / 1e4, 1),
    }
    (ROOT / "out").mkdir(exist_ok=True)
    CACHE.write_text(json.dumps(data, separators=(",", ":")))
    write(data)
    return data


CACHE = ROOT / "out" / "garden3d_data.json"


def write(data: dict) -> None:
    template = (ROOT / "viewer" / "garden3d.template.html").read_text()
    html = template.replace("/*__GARDEN_DATA__*/null", json.dumps(data, separators=(",", ":")))
    out = ROOT / "docs" / "garden3d.html"
    out.write_text(html)
    print("wrote", out, round(len(html) / 1e6, 2), "MB")


if __name__ == "__main__":
    import sys
    if "--reuse" in sys.argv and CACHE.exists():
        write(json.loads(CACHE.read_text()))
    else:
        main()
