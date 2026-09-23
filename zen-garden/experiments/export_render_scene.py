"""Export the twin as a bundle for the photoreal renderer (render/garden_cycles.py).

The renderer runs inside Blender (locally, or on a Colab GPU through the Colab CLI) and does not
import karesansui; everything it draws comes from here, computed by the model:

  - the ground at 1 mm: moss zones, stream bed, banks and the rim rule (landscape.terrain);
  - the sand at 0.5 mm after a simulated erase-and-rake cycle of one pattern;
  - water surfaces, cascades with their lips and drops, the pool;
  - stones, rocks, bridge, lanterns (their footing on the ground), the bonsai, the arm in a pose
    taken from the program's own joint trajectory;
  - stones along the water's edge where a bank stands tall, and pebbles on the stream bed
    (placed by rule, for the picture only).

Run:  python experiments/export_render_scene.py [pattern]      (writes out/render_scene/)
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import shapely
from scipy.ndimage import binary_dilation, distance_transform_edt, minimum_filter
from shapely.geometry import Polygon

from karesansui import gcode, water
from karesansui.bonsai import grow, tuft_mesh
from karesansui.config import poc_garden
from karesansui.geometry import sand_region
from karesansui.landscape import LIVING, PRESERVED, WATER, terrain
from karesansui.planner import ArmMachine, build_program
from karesansui.sim import Bed, SimCfg

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "render_scene"


def edge_stones(t, rng) -> list:
    """Stones along the water's edge: pebbles where the bank is low, stones facing it where it
    stands more than 9 mm above the water beside it (the sides of the cascades)."""
    wet = ~np.isnan(t.water)
    land = np.isin(t.zone, (LIVING, PRESERVED))
    lip = binary_dilation(wet) & ~wet & land
    low = minimum_filter(np.where(wet, t.water, np.inf), size=3)
    dx = t.x[1] - t.x[0]
    out = []
    ii, jj = np.nonzero(lip)
    for i, j in zip(ii, jj):
        step = t.ground[i, j] - low[i, j]
        tall = step > 9.0
        if rng.random() > (0.35 if tall else 0.3):           # the grid is 1 mm: thin it out
            continue
        di, dj = np.nonzero(wet[max(i - 1, 0):i + 2, max(j - 1, 0):j + 2])
        oy, ox = (di - 1).mean(), (dj - 1).mean()            # towards the water
        n = math.hypot(ox, oy) or 1.0
        x = t.x[j] + ox / n * 1.2 * dx + rng.uniform(-1, 1)
        y = t.y[i] + oy / n * 1.2 * dx + rng.uniform(-1, 1)
        if tall:
            w = min(max(step * 0.4, 3.5), 11.0)
            size = [w * rng.uniform(0.8, 1.2), w * rng.uniform(0.8, 1.2), step / 2 + 1.5]
            z = (t.ground[i, j] + low[i, j]) / 2
        else:
            w = rng.uniform(1.5, 3.8)
            size = [w * rng.uniform(0.9, 1.4), w * rng.uniform(0.9, 1.4), w * 0.5]
            z = low[i, j] + w * 0.12
        out.append([round(x, 2), round(y, 2), round(float(z), 2)] + [round(s, 2) for s in size]
                   + [round(rng.uniform(0, 2 * math.pi), 3), round(rng.uniform(0.25, 0.55), 3)])
    return out


def bed_pebbles(t, rng, n=1400) -> list:
    wet = ~np.isnan(t.water) & (t.zone == WATER)
    ii, jj = np.nonzero(wet)
    pick = rng.choice(len(ii), size=min(n, len(ii)), replace=False)
    out = []
    for k in pick:
        i, j = ii[k], jj[k]
        w = rng.uniform(1.0, 3.6)
        out.append([round(float(t.x[j] + rng.uniform(-.5, .5)), 2), round(float(t.y[i] + rng.uniform(-.5, .5)), 2),
                    round(float(t.ground[i, j] + w * 0.25), 2), round(w * rng.uniform(1, 1.5), 2), round(w, 2),
                    round(w * 0.55, 2), round(rng.uniform(0, 2 * math.pi), 3), round(rng.uniform(0.2, 0.6), 3)])
    return out


def main(pattern: str = "ripples") -> None:
    g = poc_garden()
    m = ArmMachine(g, "scara")
    W, D, S = g.tray.width, g.tray.depth, g.tray.bed_depth
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)

    # ---- ground at 1 mm
    t = terrain(g, dx=1.0, overlays=False)
    wet = ~np.isnan(t.water)
    d, (ii, jj) = distance_transform_edt(~wet, return_indices=True)
    under = np.where(d <= 3, t.water[ii, jj], np.nan)        # the surface carried 3 mm under the banks
    np.savez_compressed(OUT / "terrain.npz", ground=t.ground.astype(np.float32), zone=t.zone,
                        water=t.water.astype(np.float32), water_ext=under.astype(np.float32))

    # ---- the bonsai grown to fit the model's tree (karesansui.bonsai)
    def ground_at(x, y):
        fx = min(max(x - 0.5, 0), t.ground.shape[1] - 1.001)
        fy = min(max(y - 0.5, 0), t.ground.shape[0] - 1.001)
        j, i = int(fx), int(fy)
        u, v = fx - j, fy - i
        G = t.ground
        return float((G[i, j] * (1 - u) + G[i, j + 1] * u) * (1 - v) + (G[i + 1, j] * (1 - u) + G[i + 1, j + 1] * u) * v)
    tree_model = grow(g, ground_at)
    tv, tf = tuft_mesh()
    np.savez_compressed(OUT / "bonsai.npz",
                        branch_pts=np.concatenate([b[0] for b in tree_model.branches]).astype(np.float32),
                        branch_r=np.concatenate([b[1] for b in tree_model.branches]).astype(np.float32),
                        branch_len=np.array([len(b[0]) for b in tree_model.branches], np.int32),
                        tufts=tree_model.tuft_array.astype(np.float32), tuft_verts=tv.astype(np.float32),
                        tuft_faces=tf.astype(np.int32), base=np.array(tree_model.base, np.float32))

    # ---- the sand after a simulated cycle, at the simulation's own 0.5 mm
    cfg = SimCfg(dx=0.5)
    prog = build_program(g, pattern, machine=m)
    segs = gcode.parse(gcode.generate(prog), "scara")
    bed = Bed(g, cfg)
    samples = gcode.sample(segs, m.arm, step_mm=cfg.dx * cfg.step)
    gcode.run_on_bed(bed, samples)
    sand = sand_region(g)
    x0, y0, x1, y1 = (math.floor(sand.bounds[0]) - 3, math.floor(sand.bounds[1]) - 3,
                      math.ceil(sand.bounds[2]) + 3, math.ceil(sand.bounds[3]) + 3)
    k = int(round(1 / cfg.dx))
    rel = (bed.h - S)[y0 * k:y1 * k, x0 * k:x1 * k].astype(np.float32)
    xs = x0 + (np.arange(rel.shape[1]) + 0.5) * cfg.dx
    ys = y0 + (np.arange(rel.shape[0]) + 0.5) * cfg.dx
    mask = shapely.contains_xy(sand, *np.meshgrid(xs, ys))
    np.savez_compressed(OUT / "sand.npz", h=rel, mask=mask)

    # ---- the arm, posed at the end of the program's last groove (the pattern it has just drawn)
    q = np.array([[s.q0, s.q1] for s in segs]).reshape(-1, 4)
    released = np.repeat([not s.latched for s in segs], 2)
    raking = np.nonzero(released & (q[:, 2] < 20.0))[0]
    pose = q[raking[-1]]
    tool = m.arm.fk(pose)[0]
    elbow = m.arm.elbows(pose)[0]

    # ---- layout
    stream = next(f for f in g.features if f.kind == "stream")
    reaches, drops = water.stream_profile(stream)
    cascades = []
    for dr in drops:
        i = dr["at"]
        v, u = np.asarray(stream.outline[i], float), np.asarray(stream.outline[i - 1], float)
        into = (v - u) / np.linalg.norm(v - u)
        cascades.append({"lip": (v - into * stream.plunge).tolist(), "dir": into.tolist(),
                         "top": stream.levels[i - 1] - water.LIP_RUN, "bottom": stream.levels[i],
                         "height": dr["height_mm"]})
    X, Y = np.meshgrid(t.x, t.y)

    def footing(xy, r):
        under = (X - xy[0]) ** 2 + (Y - xy[1]) ** 2 <= r * r
        return float(t.ground[under].mean()), float(t.ground[under].min())

    lanterns = []
    for lan in g.lanterns:
        foot, low = footing(lan.xy, lan.radius)
        lanterns.append({"name": lan.name, "xy": list(lan.xy), "radius": lan.radius, "top": lan.height,
                         "light": lan.light_height, "lumens": lan.lumens, "cct": lan.cct, "foot": foot, "low": low})
    tree, pot = next(f for f in g.features if f.kind == "tree"), next(f for f in g.features if f.kind == "pot")
    pot_c = np.asarray(Polygon(pot.outline).centroid.coords[0])
    arm = g.arm
    scene = {
        "pattern": pattern,
        "tray": {"width": W, "depth": D, "bed": S, "wall": g.tray.wall_height},
        "terrain": {"dx": 1.0, "nx": len(t.x), "ny": len(t.y)},
        "sand": {"x0": x0, "y0": y0, "dx": cfg.dx, "nx": rel.shape[1], "ny": rel.shape[0],
                 "outline": [list(p) for p in g.sand_outline]},
        "stones": [{"name": s.name, "outline": [list(p) for p in s.outline], "top": s.height - S} for s in g.stones],
        "rocks": [{"name": f.name, "outline": [list(p) for p in f.outline], "top": f.height}
                  for f in g.features if f.kind == "rock"],
        "bridge": [{"outline": [list(p) for p in f.outline], "height": f.height} for f in g.features if f.kind == "bridge"],
        "stream": {"outline": [list(p) for p in stream.outline], "levels": list(stream.levels),
                   "width": stream.width, "plunge": stream.plunge},
        "cascades": cascades,
        "lanterns": lanterns,
        "tree": {"height": tree.height, "canopy": [list(p) for p in tree.outline], "pot": pot_c.tolist(),
                 "ground": float(t.ground[int(pot_c[1]), int(pot_c[0])])},
        "arm": {"base": list(arm.base), "base_radius": arm.base_radius, "l1": arm.scara.link1, "l2": arm.scara.link2,
                "link_height": arm.scara.link_height, "w1": arm.scara.link1_width, "w2": arm.scara.link2_width,
                "pose": pose.tolist(), "tool": tool.tolist(), "elbow": elbow.tolist()},
        "head": {"tines": g.rake.n_tines, "pitch": g.rake.pitch, "tine_width": g.rake.tine_width,
                 "depth": g.rake.depth, "blade": g.screed.width},
        "edge_stones": edge_stones(t, rng),
        "bed_pebbles": bed_pebbles(t, rng),
    }
    (OUT / "scene.json").write_text(json.dumps(scene))
    print(f"wrote {OUT}: terrain {t.ground.shape}, sand {rel.shape}, "
          f"{len(scene['edge_stones'])} edge stones, arm pose q={np.round(pose, 3).tolist()} at {np.round(tool[:2], 1).tolist()}")


if __name__ == "__main__":
    main(*(sys.argv[1:2] or ["ripples"]))
