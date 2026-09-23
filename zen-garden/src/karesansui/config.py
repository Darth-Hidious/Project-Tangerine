"""Garden specification: every physical parameter the twin uses, in one place.

Units are millimetres, degrees and seconds unless a field says otherwise.
Frame: origin at the inner south-west corner of the tray, x along the long
side, y along the short side, z up from the top of the base plate. The viewer
is assumed to stand on the south side (y = 0).

Values marked ``# Phase 0`` are priors from datasheets or the literature and
must be replaced by measurements before the twin's numbers are trusted.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

WALLS = ("south", "north", "west", "east")


@dataclass(frozen=True)
class Tray:
    width: float = 1200.0        # inner x extent
    depth: float = 800.0         # inner y extent
    wall_height: float = 100.0   # above the base plate
    bed_depth: float = 35.0      # nominal gravel surface above the base plate


@dataclass(frozen=True)
class Stone:
    name: str
    outline: tuple[tuple[float, float], ...]   # footprint polygon in the tray frame
    height: float                               # top of the stone above the base plate
    group: str = ""                             # stones in one group are ringed as one island
    rings: int = 1                              # rake passes around the stone (or its group)


@dataclass(frozen=True)
class Grit:
    name: str = "Edelsplitt 2/5, crushed granite"
    repose_deg: float = 37.0      # Phase 0: aggregate-industry standard for crushed stone, range 35-40
    bulk_density: float = 1500.0  # kg/m3, Phase 0
    grain_mm: float = 3.5         # median grain size; only used for the rendered grain texture
    albedo: float = 0.45          # diffuse reflectance of the grit, Phase 0 (photo vs grey card)


@dataclass(frozen=True)
class Rake:
    n_tines: int = 5
    pitch: float = 25.0
    tine_width: float = 5.0       # across the direction of travel
    depth: float = 8.0            # tine tip below the skid sole
    bar_margin: float = 3.0       # bar overhang beyond the outer tine edge
    bar_thickness: float = 12.0   # along the direction of travel, centred on the tine line
    skid_gap: float = 10.0        # tine line to the skid's trailing edge
    skid_length: float = 40.0
    skid_width: float = 30.0      # centred; a full-width skid sticks out on turns (see PLAN.md)
    hang: float = 25.0            # skid sole below the screed blade edge with the head at its lower stop
    float_travel: float = 40.0    # vertical travel of the floating head on its slide
    mode: str = "skid"            # "skid": floating head, depth set by the skid; "fixed": rigid head

    @property
    def span(self) -> float:
        return (self.n_tines - 1) * self.pitch

    @property
    def tine_offsets(self) -> tuple[float, ...]:
        mid = (self.n_tines - 1) / 2
        return tuple((k - mid) * self.pitch for k in range(self.n_tines))

    @property
    def tine_half(self) -> float:
        """Distance from the rake centre to the outer edge of the outer tine."""
        return self.span / 2 + self.tine_width / 2

    @property
    def bar_half(self) -> float:
        return self.tine_half + self.bar_margin

    @property
    def lane_spacing(self) -> float:
        """Centre-to-centre distance of adjacent passes that continue the groove pitch."""
        return self.n_tines * self.pitch

    @property
    def min_radius_hard(self) -> float:
        """Below this centreline radius the inner tine runs backwards."""
        return self.tine_half

    @property
    def min_radius_soft(self) -> float:
        """Below this the inner tine moves at under half the rake's speed and starts to pivot."""
        return 2 * self.tine_half

    @property
    def reach_ahead(self) -> float:
        """How far the skid's leading edge sits ahead of the tine line."""
        return self.skid_gap + self.skid_length


@dataclass(frozen=True)
class Screed:
    width: float = 110.0      # blade length across the direction of travel
    overlap: float = 10.0     # overlap between adjacent screed lanes
    thickness: float = 6.0    # blade thickness along the direction of travel


@dataclass(frozen=True)
class LightStrip:
    wall: str                        # south | north | west | east
    height: float = 50.0             # above the nominal gravel surface
    inset: float = 8.0               # from the wall face into the tray
    tilt: float = 0.0                # aim below horizontal, degrees
    optic: str = "bare"              # "bare" (Lambertian COB strip) or "lens" (profile with a line lens)
    beam_fwhm: float = 30.0          # vertical full width at half maximum of the lens optic, degrees
    cct: float = 2700.0              # correlated colour temperature, K
    flux_per_m: float = 1000.0       # luminous flux per metre of strip, lm/m
    lens_transmission: float = 0.9   # fraction of flux a lens profile passes

    def __post_init__(self) -> None:
        if self.wall not in WALLS:
            raise ValueError(f"unknown wall {self.wall!r}, expected one of {WALLS}")
        if self.optic not in ("bare", "lens"):
            raise ValueError(f"unknown optic {self.optic!r}")


@dataclass(frozen=True)
class Lighting:
    strips: tuple[LightStrip, ...] = tuple(LightStrip(w) for w in WALLS)
    ambient_lux: float = 10.0        # diffuse room light on the bed


@dataclass(frozen=True)
class Gantry:
    rapid_speed: float = 150.0       # mm/s
    rake_speed: float = 30.0         # Phase 0: fastest speed that still leaves clean grooves
    screed_speed: float = 40.0
    z_speed: float = 15.0
    a_speed: float = 120.0           # head rotation, deg/s
    accel: float = 400.0             # mm/s^2, XY
    junction_deviation: float = 0.05
    travel_clearance: float = 15.0   # tine tips above the nominal surface while travelling
    rake_clearance: float = 12.0     # blade edge above the nominal surface while raking
    a_limit: float | None = None     # total head rotation limit in degrees; None = slip ring


@dataclass(frozen=True)
class PlannerCfg:
    wall_clearance: float = 5.0      # minimum gap between any part of the head and a wall
    stone_clearance: float = 12.0    # minimum gap between any part of the head and a stone
    sample_step: float = 2.0         # centreline sampling for validation and G-code
    min_stroke: float = 40.0         # rake strokes shorter than this are dropped
    closed_overlap: float = 60.0     # extra travel past the start on closed rings


@dataclass(frozen=True)
class Garden:
    tray: Tray = field(default_factory=Tray)
    stones: tuple[Stone, ...] = ()
    grit: Grit = field(default_factory=Grit)
    rake: Rake = field(default_factory=Rake)
    screed: Screed = field(default_factory=Screed)
    lighting: Lighting = field(default_factory=Lighting)
    gantry: Gantry = field(default_factory=Gantry)
    planner: PlannerCfg = field(default_factory=PlannerCfg)


def _build(cls, table: dict | None):
    table = dict(table or {})
    known = set(cls.__dataclass_fields__)
    unknown = set(table) - known
    if unknown:
        raise ValueError(f"unknown keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**table)


def load_garden(path: str | Path) -> Garden:
    with open(path, "rb") as fh:
        doc = tomllib.load(fh)
    stones = tuple(
        Stone(
            name=s["name"],
            outline=tuple((float(x), float(y)) for x, y in s["outline"]),
            height=float(s["height"]),
            group=s.get("group", ""),
            rings=int(s.get("rings", 1)),
        )
        for s in doc.get("stones", [])
    )
    light_doc = dict(doc.get("lighting", {}))
    strips = tuple(_build(LightStrip, s) for s in light_doc.pop("strips", [])) or Lighting().strips
    return Garden(
        tray=_build(Tray, doc.get("tray")),
        stones=stones,
        grit=_build(Grit, doc.get("grit")),
        rake=_build(Rake, doc.get("rake")),
        screed=_build(Screed, doc.get("screed")),
        lighting=Lighting(strips=strips, **light_doc),
        gantry=_build(Gantry, doc.get("gantry")),
        planner=_build(PlannerCfg, doc.get("planner")),
    )


DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "garden.toml"


def default_garden() -> Garden:
    return load_garden(DEFAULT_CONFIG)
