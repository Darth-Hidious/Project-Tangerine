"""Bill of materials for the proof-of-concept garden, with the quantities taken from the model.

Writes docs/bom.csv (one row per line item) and BOM.md (the same, readable). Every quantity that
the model decides is computed here from configs/poc.toml and the terrain, so the BOM follows the
design when it changes: the sand, the moss areas, the fill under the land, the water, the tube, the
links. The rest are choices the plan makes (PLAN.md §4) or assumptions a CAD pass must fix; the
"basis" column says which.

Prices are estimates in euros, not quotes. Where a current listing was looked up (US listings found
by web search, September 2026, in the listing's currency), the "reference" column gives it; every
other price is a typical hobby price, unverified. Check before buying.

Run:  python experiments/bom.py
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon

from karesansui import water
from karesansui.config import poc_garden
from karesansui.landscape import BRIDGE, KERB, LIVING, PRESERVED, ROCK, WATER, terrain

ROOT = Path(__file__).resolve().parents[1]
FRAME_T, BASE_T = 22.0, 12.0          # mm: walnut board and plywood base thickness (assumed, as the renders draw them)
DRAIN = 10.0                          # mm of drainage grit under the living moss (assumed)
MARGIN = 1.15                         # buy 15 % over what the model needs, for waste and settling


def model_quantities():
    g = poc_garden()
    t = terrain(g, dx=1.0, overlays=False)
    cell = 1e-6                                           # m2 per 1 mm cell
    S = g.tray.bed_depth
    area = {n: float(np.count_nonzero(t.zone == z)) * cell
            for n, z in (("living", LIVING), ("preserved", PRESERVED), ("water", WATER))}
    land = np.isin(t.zone, [LIVING, PRESERVED, ROCK, BRIDGE, KERB])
    wet = t.zone == WATER
    gy, gx = np.gradient(t.ground, 1.0)
    surface = np.sqrt(1 + gx ** 2 + gy ** 2) * cell
    sand_m2 = Polygon(g.sand_outline).area * cell
    X, Y = np.meshgrid(t.x, t.y)
    # a lantern's own height: its top above the sand less the ground it stands on (as the tests measure it)
    own = [lan.height - float(t.ground[(X - lan.xy[0]) ** 2 + (Y - lan.xy[1]) ** 2 <= lan.radius ** 2].mean())
           for lan in g.lanterns]
    W, D = g.tray.width, g.tray.depth
    sc = g.arm.scara
    return {
        "g": g,
        "sand_l": sand_m2 * S,                                            # m2 x mm = litres
        "sand_kg": sand_m2 * S * g.grit.bulk_density / 1000.0,
        "living_m2": area["living"], "preserved_m2": area["preserved"],
        "fill_l": float(((t.ground[land] + S) * cell).sum() + ((t.ground[wet] + S) * cell).sum()),
        "drain_l": area["living"] * DRAIN,
        "water_l": float(((t.water[wet] - t.ground[wet]) * cell).sum()),
        "liner_m2": float(surface[land | wet].sum()),
        "stream_m": water.stream_length_m(g),
        "frame_m": 2 * (W + 2 * FRAME_T) + 2 * (D + 2 * FRAME_T),         # outer edge of the mitred boards
        "board_h": g.tray.wall_height + BASE_T,
        "base_mm": (W, D),
        "links_mm": (sc.link1, sc.link2, sc.link1_width, sc.link2_width),
        "column_h": sc.link_height + S + BASE_T,                           # base plate to the links' undersides
        "lift": sc.lift,
        "tube_m": g.water.tube_len_m, "tube_id": g.water.tube_id_mm,
        "flow": g.water.flow_lpm, "lift_mm": g.water.lift_mm, "reservoir_l": g.water.reservoir_l,
        "head_m": water.system_head_m(g.water.flow_lpm, g.water.lift_mm / 1000, g.water.tube_id_mm / 1000,
                                      g.water.tube_len_m),
        "lanterns": g.lanterns, "lantern_own": own,
        "rake": g.rake, "screed": g.screed,
    }


def rows(q):
    """(group, item, spec, qty, unit, basis, eur_low, eur_high, reference)."""
    L1, L2, W1, W2 = q["links_mm"]
    j1, j2 = q["g"].arm.scara.j1_limits, q["g"].arm.scara.j2_limits
    r, s = q["rake"], q["screed"]
    # to 5 mm: the footing is a mean over sloping ground, so a finer or coarser grid moves it by ~2 mm
    own = ", ".join(f"{lan.name.split()[0]} {5 * round(h / 5):.0f}" for lan, h in zip(q["lanterns"], q["lantern_own"]))
    tops = sorted({f"{lan.height:.0f}" for lan in q["lanterns"]})
    return [
        # ---- arm drives
        ("Arm drives", "Stepper motor, NEMA17", "42 mm, ~0.45 N·m holding, 1.5 A, 5 mm shaft: base (q1) and elbow (q2)",
         2, "pcs", "PLAN §4.1; the base joint needs 0.26 N·m (§3.1), 0.05 N·m at the motor through 5:1", 10, 16, "StepperOnline NEMA17 1.5 A: US$10.99–12.99"),
        ("Arm drives", "Stepper motor, NEMA17 or NEMA14", "lift (z) and tool yaw (q4)", 2, "pcs", "PLAN §4.1", 9, 16, ""),
        ("Arm drives", "GT2 pulley 16T, 6 mm belt, 5 mm bore", "motor pulleys for the 5:1 base and elbow drives",
         2, "pcs", "PLAN §4.1 asks 20T:100T; 16T:80T is the same 5:1 from common sizes (100T is rarely stocked)", 2, 4, ""),
        ("Arm drives", "GT2 pulley 80T, 6 mm belt", "joint pulleys, bore to suit the joint shaft (CAD)",
         2, "pcs", "5:1 with the 16T", 6, 15, ""),
        ("Arm drives", "GT2 pulleys 20T + 60T, 6 mm belt", "tool yaw, 3:1", 1, "set", "PLAN §4.1", 5, 10, ""),
        ("Arm drives", "GT2 closed belts, 6 mm", "lengths from the pulley centre distances (CAD); the elbow belt runs through link 1",
         3, "pcs", "PLAN §4.1", 3, 8, ""),
        ("Arm drives", "Lead screw T8 + brass nut", f"{q['lift'] + 40:.0f} mm long for {q['lift']:.0f} mm of stroke, with a coupler",
         1, "set", "PLAN §4.1 (a belt is the alternative)", 8, 15, ""),
        ("Arm drives", "Linear guide for the lift", "e.g. MGN9 rail 150 mm + carriage, or two 8 mm rods + bushings",
         1, "set", "Not in the plan yet: the lift needs a guide (CAD)", 12, 30, ""),
        ("Arm drives", "Bearings, preloaded pairs", "base, elbow, lift and yaw joints; sizes from the shafts (CAD)",
         8, "pcs", "PLAN §4.1 (a pair per joint)", 2, 8, ""),
        ("Arm drives", "Hard stops", "pins or blocks at q1 {:.0f}°/{:.0f}° and q2 {:.0f}°/{:.0f}°".format(*j1, *j2), 4, "pcs",
         "PLAN §4.1, configs/poc.toml joint limits", 1, 3, ""),
        # ---- arm structure
        ("Arm structure", "Walnut for the links", f"link 1: {L1:.0f} mm between axes, {W1:.0f} mm wide; link 2: {L2:.0f} mm, {W2:.0f} mm wide",
         2, "pcs", "configs/poc.toml [arm.scara]", 8, 20, ""),
        ("Arm structure", "Aluminium flat bar", f"laminated under each link, ~{max(L1, L2) + max(W1, W2):.0f} mm long; section from a stiffness check (not done)",
         2, "pcs", "PLAN §4.1; the twin assumes rigid links (§8)", 4, 10, ""),
        ("Arm structure", "Column", f"walnut, Ø{2 * q['g'].arm.base_radius:.0f} mm, ~{q['column_h']:.0f} mm from the base plate to the links",
         1, "pcs", "configs/poc.toml base_radius and link_height", 15, 40, ""),
        ("Arm structure", "Hubs and motor mounts", "3D-printed or turned", 1, "set", "PLAN §5", 10, 25, ""),
        ("Arm structure", "Stainless fasteners", "M3/M4/M5 assortment", 1, "set", "PLAN §4.3", 8, 15, ""),
        # ---- head
        ("Head", "Vertical slide", f"{r.float_travel:.0f} mm travel, for the floating comb", 1, "pcs", "PLAN §3.2, [rake] float_travel", 8, 20, ""),
        ("Head", "Stainless pins", f"Ø{r.tine_width:.1f} mm, {r.n_tines} at {r.pitch:.0f} mm pitch, tips {r.depth:.0f} mm below the skid",
         r.n_tines, "pcs", "[rake]", 0.5, 1.5, ""),
        ("Head", "Skid", f"POM or brass, {r.skid_length:.0f} × {r.skid_width:.0f} mm", 1, "pcs", "[rake]", 2, 6, ""),
        ("Head", "Screed blade", f"{s.width:.0f} × {s.thickness:.0f} mm, stainless or brass", 1, "pcs", "[screed]", 2, 6, ""),
        ("Head", "Micro servo", "latch for the slide, on FluidNC's analog output", 1, "pcs", "PLAN §3.2", 3, 8, ""),
        ("Head", "Capsule slip ring", "6 circuits, 12.5 mm, for the servo's three wires", 1, "pcs", "PLAN §4.1",
         4, 10, "eBay 6-circuit 12.5 mm 2 A: €4.06"),
        ("Head", "Hall sensor + magnet", "yaw index", 1, "set", "PLAN §4.1", 2, 5, ""),
        # ---- control
        ("Control", "FluidNC controller", "ESP32 with ≥ 4 stepper drivers, e.g. V1 Engineering Jackpot3 (6 × TMC2226)",
         1, "pcs", "PLAN §4.2", 70, 110, "Jackpot3: US$76.99"),
        ("Control", "24 V power adapter", "external, certified (CE/UL), 60–100 W", 1, "pcs", "PLAN §4.2", 15, 35, ""),
        ("Control", "Buck converter 24 → 12 V", "≥ 3 A: pump, lantern LEDs, servo", 1, "pcs", "PLAN §4.2", 3, 10, ""),
        ("Control", "Limit switches", "homing q1, q2, z", 3, "pcs", "PLAN §4.1", 1, 3, ""),
        ("Control", "Wire, connectors, drag chain, drip loops", "", 1, "set", "PLAN §4.2", 15, 35, ""),
        # ---- water
        ("Water", "Pump, 12 V DC brushless", f"≥ {q['head_m']:.2f} m head at {q['flow']:.1f} L/min, with a flow adjuster or PWM",
         1, "pcs", "water.py operating point (PLAN §3.7)", 8, 20, "12 V 3 m / 240 L/h brushless: US$9.75–12.61"),
        ("Water", "Reservoir", f"{q['reservoir_l']:.0f} L, dark and lidded, 224 × 162 × 85 mm under the pool", 1, "pcs",
         "configs/poc.toml [water]", 5, 15, ""),
        ("Water", "Silicone tube", f"{q['tube_id']:.0f} mm ID, {q['tube_m'] * MARGIN:.1f} m", 1, "pcs", "[water] tube_len_m + 15 %", 3, 8, ""),
        ("Water", "Float switch", "cuts the pump's power directly", 1, "pcs", "PLAN §4.2", 3, 8, ""),
        ("Water", "Leak sensor", "drawer floor", 1, "pcs", "PLAN §4.2", 3, 10, ""),
        ("Water", "Sponge pre-filter", "pump intake", 1, "pcs", "PLAN §4.3", 2, 5, ""),
        ("Water", "EPDM liner", f"{q['liner_m2'] * MARGIN:.2f} m² of surface (land and water, slopes included) plus laps up the walls: one 1 × 1 m sheet",
         1, "pcs", "terrain surface area", 8, 20, ""),
        # ---- garden
        ("Garden", "White quartz sand, washed", f"0.1–0.5 mm; the bed holds {q['sand_l']:.2f} L = {q['sand_kg']:.2f} kg at 1450 kg/m³",
         math.ceil(q["sand_kg"] * MARGIN), "kg", "sand outline × 25 mm bed", 1, 3, ""),
        ("Garden", "Living moss", f"{q['living_m2'] * 1e4:.0f} cm² (within 45 mm of the water)", round(q["living_m2"] * MARGIN, 3), "m²",
         "terrain zones", 300, 900, ""),
        ("Garden", "Preserved moss", f"{q['preserved_m2'] * 1e4:.0f} cm² (the dry ground)", round(q["preserved_m2"] * MARGIN, 3), "m²",
         "terrain zones", 100, 400, ""),
        ("Garden", "Fill under the land", f"{q['fill_l']:.1f} L from the base plate up to the ground: substrate on the living side "
         "(the hill is 65 mm high), a foam or cork former can replace it under the preserved moss", round(q["fill_l"] * MARGIN, 1), "L",
         "terrain heights", 0.5, 1.5, ""),
        ("Garden", "Drainage grit", f"{DRAIN:.0f} mm under the living moss", round(q["drain_l"] * MARGIN, 2), "L",
         "living area × 10 mm (assumed)", 2, 5, ""),
        ("Garden", "Stones", "the pair in the sand: 70 and 30 mm above it", 2, "pcs", "configs/poc.toml [[stones]]", 3, 15, ""),
        ("Garden", "Rocks", "spring rock 90 mm (sealed into the liner), cascade rocks 48, 30, 18 mm", 4, "pcs", "configs/poc.toml features", 3, 15, ""),
        ("Garden", "Stream-edge stones and bed pebbles", f"10–20 mm stones and gravel for the {q['stream_m'] * 1000:.0f} mm stream", 1, "kg",
         "stream length", 3, 10, ""),
        ("Garden", "Stone lanterns", f"about {own} mm tall on their footings (tops {'/'.join(tops)} mm above the sand), hollow for an LED",
         len(q["lanterns"]), "pcs",
         "configs/poc.toml [[lanterns]]", 8, 25, ""),
        ("Garden", "Warm LEDs, 2200 K", "~60 lm each at 12 V", len(q["lanterns"]), "pcs", "configs/poc.toml lumens, cct", 1, 4, ""),
        ("Garden", "Grow light", "dimmable, on a timer, over the living side and the tree", 1, "pcs", "PLAN §3.6", 20, 60, ""),
        ("Garden", "Bonsai", "an indoor species, ≤ 300 mm tall with the model's canopy (§3.8)", 1, "pcs", "your own tree", 0, 0, ""),
        # ---- tray
        ("Tray", "Walnut boards for the frame", f"{FRAME_T:.0f} × {q['board_h']:.0f} mm, {q['frame_m'] / 1000 * MARGIN:.1f} m for four mitred boards",
         round(q["frame_m"] / 1000 * MARGIN, 1), "m", f"tray {q['base_mm'][0]:.0f} × {q['base_mm'][1]:.0f} mm; {FRAME_T:.0f} mm thickness assumed", 10, 25, ""),
        ("Tray", "Plywood base", f"{BASE_T:.0f} mm, {q['base_mm'][0]:.0f} × {q['base_mm'][1]:.0f} mm inside the frame", 1, "pcs",
         "configs/poc.toml [tray]", 10, 25, ""),
        ("Tray", "Sand basin", "sealed basin with a 5–10 mm kerb, to the sand outline", 1, "pcs", "PLAN §4.3", 10, 30, ""),
        ("Tray", "Epoxy or sealant, and oil finish", "", 1, "set", "PLAN §4.3", 15, 35, ""),
        ("Tray", "Electronics drawer and plinth", "under the dry right-hand side, above the highest water", 1, "set", "PLAN §4.2", 20, 50, ""),
    ]


def main() -> None:
    q = model_quantities()
    items = rows(q)
    header = ["group", "item", "spec", "qty", "unit", "basis", "eur_low", "eur_high", "reference"]
    out_csv = ROOT / "docs" / "bom.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for it in items:
            w.writerow(it)

    def cost(it, k):   # the price per unit (per piece, kg, m², L or m) times the quantity
        return (it[6] if k == 0 else it[7]) * it[3]

    lines = ["# Bill of materials",
             "",
             "For the proof-of-concept garden in [PLAN.md](PLAN.md). Generated by `experiments/bom.py` from "
             "`configs/poc.toml` and the terrain model, so the quantities follow the design. Also as "
             "[docs/bom.csv](docs/bom.csv).",
             "",
             "**Prices are estimates, not quotes.** The euro columns are typical hobby prices per unit, unverified. "
             "Where a current listing was looked up (US listings found by web search, September 2026), the reference "
             "column gives it in the listing's currency. Check before buying.",
             "",
             "**Basis** says where each line comes from: a config value, a model output, a choice in PLAN.md §4, or an "
             "assumption that a CAD pass must fix.",
             ""]
    groups = []
    for it in items:
        if it[0] not in groups:
            groups.append(it[0])
    total_lo = total_hi = 0.0
    for grp in groups:
        sel = [it for it in items if it[0] == grp]
        lo, hi = sum(cost(it, 0) for it in sel), sum(cost(it, 1) for it in sel)
        total_lo, total_hi = total_lo + lo, total_hi + hi
        lines += [f"## {grp} (≈ €{lo:.0f}–{hi:.0f})", "",
                  "| Item | Spec | Qty | € per unit of qty | Basis | Reference |", "|---|---|---|---|---|---|"]
        for _, item, spec, qty, unit, basis, lo_u, hi_u, ref in sel:
            price = "own" if hi_u == 0 else f"{lo_u:g}–{hi_u:g}"
            lines.append(f"| {item} | {spec} | {qty:g} {unit} | {price} | {basis} | {ref} |")
        lines.append("")
    lines += [f"**Total ≈ €{total_lo:.0f}–{total_hi:.0f}**, before shipping. The wood and the moss make most of the spread.", "",
              "## Model quantities behind it", "",
              f"- Sand bed: {q['sand_l']:.2f} L, {q['sand_kg']:.2f} kg.",
              f"- Living moss {q['living_m2'] * 1e4:.0f} cm², preserved moss {q['preserved_m2'] * 1e4:.0f} cm², open water in the stream and pools "
              f"{q['water_l']:.2f} L standing.",
              f"- Fill from the base plate up to the ground: {q['fill_l']:.1f} L. Liner surface over land and water: {q['liner_m2']:.3f} m².",
              f"- Pump duty: {q['head_m']:.2f} m of head at {q['flow']:.1f} L/min ({q['lift_mm']:.0f} mm lift, "
              f"{q['tube_m']:.1f} m of {q['tube_id']:.0f} mm tube).",
              f"- Buying margin: {100 * (MARGIN - 1):.0f} % on bulk quantities.",
              ""]
    (ROOT / "BOM.md").write_text("\n".join(lines))
    print(f"wrote {out_csv} and BOM.md: {len(items)} lines, total ≈ €{total_lo:.0f}–{total_hi:.0f}")


if __name__ == "__main__":
    main()
