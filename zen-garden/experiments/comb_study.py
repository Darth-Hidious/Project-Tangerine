"""How wide a comb, and where the stones go, for four patterns that actually look different.

On a 9 dm2 bed the border pass (the frame) and the ring around the stone pair are needed by
every pattern, and together they can take almost all of the raking, which leaves lines, waves
and spirals only scraps. For each variant this plans every pattern and reports the share of
the raked length that belongs to the pattern itself, plus rake and erase coverage.

Run:  python experiments/comb_study.py      (seconds; writes docs/comb_study.json)
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from karesansui.config import poc_garden
from karesansui.geometry import cumulative_length
from karesansui.planner import ArmMachine, build_program

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = ["lines", "waves", "ripples", "spiral"]


def comb(g, n_tines: int, blade: float, overlap: float):
    return dataclasses.replace(g, rake=dataclasses.replace(g.rake, n_tines=n_tines),
                               screed=dataclasses.replace(g.screed, width=blade, overlap=overlap))


def moved(g, dx: float, dy: float):
    return dataclasses.replace(g, stones=tuple(
        dataclasses.replace(s, outline=tuple((x + dx, y + dy) for x, y in s.outline)) for s in g.stones))


def main() -> dict:
    base = poc_garden()                                          # 5 tines, 44 mm blade
    nine = comb(base, 9, 76.0, 8.0)                              # the first proof-of-concept head
    variants = {
        "9 tines, 76 mm blade (first design)": nine,
        "9 tines, stones moved back-right": moved(nine, 45, 60),
        "5 tines, 44 mm blade (chosen)": base,
        "5 tines, stones moved back-right": moved(base, 45, 60),
    }
    out = {}
    for name, g in variants.items():
        m = ArmMachine(g, "scara")
        rows = {}
        for pat in PATTERNS:
            prog = build_program(g, pat, machine=m)
            r = prog.report
            length = {}
            for p in prog.passes:
                if p.kind == "rake":
                    kind = p.label.split(":")[0]
                    length[kind] = length.get(kind, 0.0) + float(cumulative_length(p.poses[:, :2])[-1])
            total = sum(length.values())
            own = sum(v for k, v in length.items() if k not in ("ring", "frame"))
            rows[pat] = {"ok": r.ok, "collisions": r.collisions, "pattern_share": round(own / total, 3),
                         "rake_m_by_kind": {k: round(v / 1000, 3) for k, v in sorted(length.items())},
                         "coverage": round(r.coverage, 3), "erase_coverage": round(r.erase_coverage, 3),
                         "min_radius_mm": round(r.min_radius_mm, 1)}
        out[name] = rows
    (ROOT / "docs" / "comb_study.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    for name, rows in main().items():
        print(name)
        for pat, r in rows.items():
            print(f"  {pat:8s} pattern share {r['pattern_share']:.0%}  coverage {r['coverage']:.0%}  "
                  f"erase {r['erase_coverage']:.0%}  collisions {r['collisions']}")
