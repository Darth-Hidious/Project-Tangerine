"""Water loop sizing for the stream: evaporation, reservoir autonomy, pump duty, stream depth.

Sources
  Evaporation: ASHRAE HVAC Applications (natatoriums), w = A (p_w - p_a)(0.089 + 0.0782 V) / Y,
    w in kg/s, A in m^2, vapour pressures in kPa, V air speed in m/s, Y latent heat in kJ/kg.
    It includes splashing for a pool of "normal activity"; a babbling stream is taken at
    activity 1.0, still water at 0.5 (the ASHRAE activity-factor convention).
  Saturation vapour pressure: Magnus form, Alduchov & Eskridge (1996), good to ~0.3 % at 0-50 C.
  Pipe losses: Darcy-Weisbach; 64/Re when laminar, Blasius (0.316 Re^-0.25) when turbulent (smooth
    tube, Re < 1e5).
  Stream depth: Nusselt film on an incline, q = g S h^3 / (3 nu) per unit width. That is a
    smooth-bed lower bound; pebbles and cascades make the real stream deeper and slower.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union

from .config import Garden
from .geometry import sand_region

MOSS_BAND = 45.0     # living moss grows within this distance of open water ...
DRY_GAP = 30.0       # ... but no closer than this to the raked sand, so damp never reaches it

G = 9.81
RHO = 998.0          # kg/m^3 at 20 C
NU = 1.0e-6          # m^2/s kinematic viscosity at 20 C


def p_sat_kpa(t_c: float) -> float:
    return 0.61094 * math.exp(17.625 * t_c / (t_c + 243.04))


def latent_heat_kj_kg(t_c: float) -> float:
    return 2501.0 - 2.361 * t_c


def evaporation_l_per_day(area_m2: float, t_water: float = 20.0, t_air: float = 22.0, rh: float = 0.45,
                          v_air: float = 0.1, activity: float = 1.0) -> float:
    p_w = p_sat_kpa(t_water)
    p_a = rh * p_sat_kpa(t_air)
    w = area_m2 * max(p_w - p_a, 0.0) * (0.089 + 0.0782 * v_air) / latent_heat_kj_kg(t_water) * activity
    return w * 86400.0 / RHO * 1000.0


CASCADE = 5.0        # a level drop of at least this much at a stream vertex is a cascade (mm)
LIP_RUN = 2.0        # a reach ending in a cascade falls this much before its lip (mm)


def cascades(stream) -> list[int]:
    """Indices of the stream vertices where the water falls as a cascade."""
    lv = stream.levels
    return [i for i in range(1, len(lv)) if lv[i - 1] - lv[i] >= CASCADE]


def water_zones(garden: Garden):
    shapes = []
    for f in garden.features:
        if f.kind == "stream":
            shapes.append(LineString(f.outline).buffer(f.width / 2))
            if f.levels and f.plunge > 0:
                shapes += [Point(f.outline[i]).buffer(f.plunge) for i in cascades(f)]
        elif f.kind == "pool":
            shapes.append(Polygon(f.outline))
    return unary_union(shapes) if shapes else Polygon()


def stream_profile(stream) -> tuple[list[dict], list[dict]]:
    """Reaches and cascades of a stream with ``levels``.

    A reach runs between two centre-line vertices. If the downstream vertex is a cascade, the
    reach falls LIP_RUN to its lip and the water then drops freely to the next level; otherwise
    the reach slopes straight to the next level."""
    pts, lv = stream.outline, stream.levels
    falls = set(cascades(stream))
    reaches, drops = [], []
    for i in range(len(pts) - 1):
        length = LineString(pts[i:i + 2]).length
        end = lv[i] - LIP_RUN if i + 1 in falls else lv[i + 1]
        reaches.append({"from": i, "length_mm": length, "fall_mm": lv[i] - end})
        if i + 1 in falls:
            drops.append({"at": i + 1, "xy": pts[i + 1], "height_mm": end - lv[i + 1]})
    return reaches, drops


def living_moss_zone(garden: Garden, band: float = MOSS_BAND, dry_gap: float = DRY_GAP):
    """Where the living (wet) moss goes: within ``band`` of open water, inside the tray, off the
    sand and at least ``dry_gap`` from it. Everything else that is not sand, water or rock is
    preserved (dry) moss."""
    water = water_zones(garden)
    if water.is_empty:
        return Polygon()
    tray = box(0.0, 0.0, garden.tray.width, garden.tray.depth)
    return (water.buffer(band).intersection(tray).difference(water)
            .difference(sand_region(garden).buffer(dry_gap, quad_segs=64)))


def open_water_m2(garden: Garden) -> float:
    if garden.water is not None and garden.water.area_m2 > 0:
        return garden.water.area_m2
    return water_zones(garden).area / 1e6


def stream_length_m(garden: Garden) -> float:
    return sum(LineString(f.outline).length for f in garden.features if f.kind == "stream") / 1000.0


def friction_factor(re: float) -> float:
    if re <= 0:
        return 0.0
    return 64.0 / re if re < 2300 else 0.316 * re ** -0.25


def system_head_m(flow_lpm: float, lift_m: float, tube_id_m: float, tube_len_m: float, k_minor: float = 3.0) -> float:
    q = flow_lpm / 60000.0
    area = math.pi * tube_id_m**2 / 4
    v = q / area
    re = v * tube_id_m / NU
    dyn = v**2 / (2 * G)
    return lift_m + friction_factor(re) * tube_len_m / tube_id_m * dyn + k_minor * dyn


def operating_point(h_max_m: float, q_max_lpm: float, lift_m: float, tube_id_m: float, tube_len_m: float,
                    k_minor: float = 3.0) -> tuple[float, float]:
    """Flow (L/min) and head (m) where a linear pump curve H = h_max (1 - Q/q_max) meets the system."""
    lo, hi = 0.0, q_max_lpm
    if system_head_m(0.0, lift_m, tube_id_m, tube_len_m, k_minor) >= h_max_m:
        return 0.0, lift_m
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if h_max_m * (1 - mid / q_max_lpm) > system_head_m(mid, lift_m, tube_id_m, tube_len_m, k_minor):
            lo = mid
        else:
            hi = mid
    q = 0.5 * (lo + hi)
    return q, h_max_m * (1 - q / q_max_lpm)


def film_depth_mm(flow_lpm: float, width_m: float, slope: float) -> tuple[float, float, float]:
    """Smooth-bed laminar film: depth (mm), mean speed (m/s), Froude number."""
    q = flow_lpm / 60000.0 / width_m
    h = (3 * NU * q / (G * slope)) ** (1 / 3)
    v = q / h
    return h * 1000.0, v, v / math.sqrt(G * h)


@dataclass
class WaterReport:
    open_water_m2: float
    wet_moss_m2: float
    evaporation_l_day: tuple[float, float]         # open water only, open water + wet moss
    autonomy_days: tuple[float, float]
    system_head_m: float
    hydraulic_power_w: float
    stream_length_m: float
    reaches: list                                  # per reach: length, fall, smooth-bed film depth, speed, Froude
    cascades: list                                 # per cascade: free-fall height, speed at the plunge pool


def report(garden: Garden) -> WaterReport:
    w = garden.water
    area = open_water_m2(garden)
    moss = living_moss_zone(garden).area / 1e6
    e_water = evaporation_l_per_day(area, activity=1.0)
    e_both = e_water + evaporation_l_per_day(0.5 * moss, activity=0.5)       # moss wet about half the time
    usable = w.reservoir_l * w.usable_fraction
    head = system_head_m(w.flow_lpm, w.lift_mm / 1000.0, w.tube_id_mm / 1000.0, w.tube_len_m)
    power = RHO * G * (w.flow_lpm / 60000.0) * head
    stream = next(f for f in garden.features if f.kind == "stream")
    reaches, drops = stream_profile(stream)
    width = stream.width / 1000.0
    speed_at = {}
    for r in reaches:
        slope = max(r["fall_mm"], 1e-3) / r["length_mm"]
        h, v, fr = film_depth_mm(w.flow_lpm, width, slope)
        r.update({"slope": round(slope, 4), "film_mm": round(h, 2), "speed_ms": round(v, 3), "froude": round(fr, 2)})
        speed_at[r["from"] + 1] = v
    for d in drops:
        fall = math.sqrt(2 * G * d["height_mm"] / 1000.0)
        d.update({"freefall_ms": round(fall, 3),
                  "impact_ms": round(math.hypot(fall, speed_at.get(d["at"], 0.0)), 3)})
    return WaterReport(area, moss, (e_water, e_both), (usable / e_both, usable / e_water), head, power,
                       stream_length_m(garden), reaches, drops)
