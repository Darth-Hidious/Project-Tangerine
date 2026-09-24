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
    wall_height: float = 100.0   # above the base plate, the same all round
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
    feather: float = 30.0     # the blade rises over the last this-many mm of a pass ...
    feather_rise: float = 3.0 # ... by this much, spreading the sand it carries instead of dropping a pile


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
class Feature:
    """A non-sand element of the layout. Drawn, and an obstacle for the arm if it is tall.

    kind: stream | pool | rock | bridge | tree | hill | pot | bay | reservoir
      stream  outline is the centre line; ``levels`` gives the water surface at each vertex (mm
              above the sand). Where it falls by 5 mm or more the water drops as a cascade into a
              plunge pool of radius ``plunge``.
      pool    ``height`` is its water surface.
      tree    outline is the canopy footprint, ``height`` the top of the foliage.
      hill    the ground rises to ``height`` inside the outline (smoothly, from its edge inwards).
      pot     a pot sunk into the ground (hidden); not an obstacle.
    """
    name: str
    kind: str
    outline: tuple[tuple[float, float], ...]    # polygon, or the centre line of a stream
    height: float = 0.0                         # top above the sand surface
    width: float = 0.0                          # stream width (centre-line features only)
    levels: tuple[float, ...] = ()              # stream water surface per centre-line vertex
    plunge: float = 0.0                         # radius of the plunge pool below each cascade


@dataclass(frozen=True)
class Lantern:
    name: str
    xy: tuple[float, float]
    radius: float = 28.0          # footprint
    height: float = 120.0         # overall height above the sand surface
    light_height: float = 70.0    # LED in the firebox, above the sand surface
    lumens: float = 60.0
    cct: float = 2200.0


@dataclass(frozen=True)
class ScaraCfg:
    """Two horizontal links, a vertical lift at the elbow end, and a yaw joint for the rake head."""
    link1: float = 230.0
    link2: float = 230.0
    link_height: float = 175.0    # underside of the links above the sand surface
    link1_width: float = 52.0     # plan-view widths of the links, for the sweep check and the renders
    link2_width: float = 42.0
    lift: float = 110.0           # vertical stroke of the tool carriage
    j1_limits: tuple[float, float] = (-170.0, 170.0)
    j2_limits: tuple[float, float] = (-150.0, 150.0)
    link1_mass: float = 0.40      # kg, for torque estimates
    link2_mass: float = 0.30
    tool_mass: float = 0.35       # lift carriage + yaw motor + rake head


@dataclass(frozen=True)
class ArticulatedCfg:
    """Base yaw, shoulder and elbow pitch, a wrist pitch that keeps the tool vertical, wrist yaw."""
    shoulder_height: float = 150.0   # shoulder axis above the sand surface
    upper: float = 250.0
    fore: float = 250.0
    wrist_drop: float = 90.0         # wrist pitch axis above the skid sole
    j1_limits: tuple[float, float] = (-170.0, 170.0)
    j2_limits: tuple[float, float] = (-10.0, 120.0)    # shoulder, from horizontal, up positive
    j3_limits: tuple[float, float] = (-160.0, 0.0)     # elbow, relative to the upper arm
    upper_mass: float = 0.45
    fore_mass: float = 0.35
    tool_mass: float = 0.35


@dataclass(frozen=True)
class Arm:
    kind: str = "scara"                          # "scara" | "articulated"
    base: tuple[float, float] = (640.0, 225.0)
    base_radius: float = 45.0
    zero_deg: float = 180.0                      # tray-frame direction of base-yaw zero (the middle of its range)
    scara: ScaraCfg = field(default_factory=ScaraCfg)
    articulated: ArticulatedCfg = field(default_factory=ArticulatedCfg)
    travel_lift: float = 85.0        # skid sole above the sand surface while travelling


@dataclass(frozen=True)
class Water:
    area_m2: float = 0.0             # open water surface; 0 = measure it from the stream and pool
    reservoir_l: float = 2.0
    usable_fraction: float = 0.5     # share of the reservoir the pump can draw before it runs dry
    flow_lpm: float = 1.0            # pumped flow
    lift_mm: float = 80.0            # reservoir surface to the stream source
    tube_id_mm: float = 6.0
    tube_len_m: float = 0.7


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
    sand_outline: tuple[tuple[float, float], ...] = ()   # empty: the whole tray is sand
    features: tuple[Feature, ...] = ()
    lanterns: tuple[Lantern, ...] = ()
    arm: Arm | None = None                               # None: the gantry machine
    water: Water | None = None


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
    if "strips" in light_doc:                    # an explicit empty list means: no rim strips
        strips = tuple(_build(LightStrip, s) for s in light_doc.pop("strips"))
    else:
        strips = Lighting().strips
    arm = None
    if "arm" in doc:
        arm_doc = dict(doc["arm"])
        scara = _build(ScaraCfg, _tuples(arm_doc.pop("scara", {})))
        artic = _build(ArticulatedCfg, _tuples(arm_doc.pop("articulated", {})))
        arm = Arm(scara=scara, articulated=artic, **_tuples(arm_doc))
    features = tuple(
        Feature(name=f["name"], kind=f["kind"], outline=tuple((float(x), float(y)) for x, y in f["outline"]),
                height=float(f.get("height", 0.0)), width=float(f.get("width", 0.0)),
                levels=tuple(float(v) for v in f.get("levels", ())), plunge=float(f.get("plunge", 0.0)))
        for f in doc.get("features", [])
    )
    return Garden(
        tray=_build(Tray, doc.get("tray")),
        stones=stones,
        grit=_build(Grit, doc.get("grit")),
        rake=_build(Rake, doc.get("rake")),
        screed=_build(Screed, doc.get("screed")),
        lighting=Lighting(strips=strips, **light_doc),
        gantry=_build(Gantry, doc.get("gantry")),
        planner=_build(PlannerCfg, doc.get("planner")),
        sand_outline=tuple((float(x), float(y)) for x, y in doc.get("sand", {}).get("outline", [])),
        features=features,
        lanterns=tuple(_build(Lantern, _tuples(l)) for l in doc.get("lanterns", [])),
        arm=arm,
        water=_build(Water, doc["water"]) if "water" in doc else None,
    )


def _tuples(table: dict) -> dict:
    """TOML arrays arrive as lists; frozen dataclasses want tuples."""
    return {k: tuple(v) if isinstance(v, list) else v for k, v in dict(table).items()}


CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
DEFAULT_CONFIG = CONFIG_DIR / "garden.toml"      # the first study: a 120 x 80 cm gravel tray and a gantry
POC_CONFIG = CONFIG_DIR / "poc.toml"             # the proof of concept: a 70 x 45 cm table garden with an arm


def default_garden() -> Garden:
    return load_garden(DEFAULT_CONFIG)


def poc_garden() -> Garden:
    return load_garden(POC_CONFIG)
