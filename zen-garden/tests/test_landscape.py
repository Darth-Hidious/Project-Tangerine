"""The ground around the sand: every bank holds its water, and everything stays inside a frame
that is one height all round."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.ndimage import binary_dilation, distance_transform_edt, maximum_filter

from karesansui.config import poc_garden
from karesansui.landscape import (BANK, CUSHION, KERB, KERB_HEIGHT, KERB_WIDTH, LIVING, PRESERVED, RIM_FREEBOARD,
                                  RIM_SLOPE, ROCK, SAND, SAND_SLOPE, WATER, bank_heights, terrain)


@pytest.fixture(scope="module")
def ground():
    g = poc_garden()
    return g, terrain(g, dx=2.0, overlays=False)


def bank_shortfall(g, t) -> float:
    """How far the lowest bank falls short of standing BANK above the water beside it (<= 0: all hold).
    A bank is the ground, or a rock standing in it (set into the liner, it holds water to its top)."""
    wet = ~np.isnan(t.water)
    bank = binary_dilation(wet) & ~wet & np.isin(t.zone, (LIVING, PRESERVED, KERB, ROCK))
    highest_neighbour = maximum_filter(np.where(wet, t.water, -np.inf), size=3)
    assert bank.sum() > 300
    return float(np.max(highest_neighbour[bank] + BANK - bank_heights(g, t)[bank]))


def test_no_water_stands_above_its_banks(ground):
    g, t = ground
    wet = ~np.isnan(t.water)
    assert bank_shortfall(g, t) <= 0.0
    assert np.all(t.ground[wet] < t.water[wet])                               # the bed is under the water


def test_the_ground_stays_inside_a_frame_of_one_height(ground):
    g, t = ground
    X, Y = np.meshgrid(t.x, t.y)
    land = np.isin(t.zone, (LIVING, PRESERVED, KERB))
    rim = g.tray.wall_height - g.tray.bed_depth
    d_wall = np.minimum.reduce([X, g.tray.width - X, Y, g.tray.depth - Y])
    at_walls = land & (d_wall <= 2.0)
    assert at_walls.sum() > 500                                               # land runs along every wall
    assert t.ground[at_walls].max() <= rim - RIM_FREEBOARD + RIM_SLOPE * 2.0
    assert np.all(t.ground[land] <= rim - RIM_FREEBOARD + RIM_SLOPE * d_wall[land] + 1e-9)
    assert t.ground[land].max() > rim + 15                                    # the hill still rises above it inside


def test_the_frame_is_as_low_as_the_spring_allows(ground):
    # Negative controls: 5 mm lower and the spring's pool overflows its left side, where the spring
    # rock ends; and with the spring rock as first drawn (36 x 26 mm, not reaching round the pool)
    # the same 55 mm frame leaves the bank behind the spring about 12 mm short.
    g, _ = ground
    lower = replace(g, tray=replace(g.tray, wall_height=g.tray.wall_height - 5.0))
    assert bank_shortfall(lower, terrain(lower, dx=2.0, overlays=False)) > 3.0
    small = ((136, 416), (150, 406), (168, 410), (172, 426), (150, 432))
    first = replace(g, features=tuple(replace(f, outline=small) if f.name == "spring rock" else f for f in g.features))
    assert bank_shortfall(first, terrain(first, dx=2.0, overlays=False)) > 10.0


def lantern_fits(t, lan) -> bool:
    """A lantern's heights are above the sand (what the arm must clear), but it stands on the moss:
    from its footing up it must be tall enough to be a lantern, with the LED in the firebox, well
    above its foot."""
    X, Y = np.meshgrid(t.x, t.y)
    foot = t.ground[(X - lan.xy[0]) ** 2 + (Y - lan.xy[1]) ** 2 <= lan.radius ** 2].mean()
    body = lan.height - foot
    return body >= 50.0 and lan.light_height - foot >= 0.45 * body


def test_every_lantern_stands_on_its_ground(ground):
    g, t = ground
    for lan in g.lanterns:
        assert lantern_fits(t, lan), lan.name
        assert lan.height <= g.arm.travel_lift - 5.0, lan.name
    # Negative control: the spring lantern as first specified (top 80 mm, LED 48 mm above the
    # sand) stands on the hill with its LED 14 mm above its foot, and fails
    spring = next(lan for lan in g.lanterns if lan.name == "spring lantern")
    assert not lantern_fits(t, replace(spring, height=80.0, light_height=48.0))


def test_no_cliff_at_the_sand(ground):
    # Beside the upper stream the banks must stand ~50 mm high only ~50 mm from the sand; the ground
    # has to come down to the kerb as a slope (at most SAND_SLOPE, plus the moss cushions), not a wall
    g, t = ground
    sand = t.zone == SAND
    d = distance_transform_edt(~sand) * (t.x[1] - t.x[0])
    land = np.isin(t.zone, (LIVING, PRESERVED, KERB))
    limit = KERB_HEIGHT + SAND_SLOPE * np.maximum(d - KERB_WIDTH, 0.0) + CUSHION + 1e-6
    dry = land & (d < 40.0)                  # the water keeps 40 mm off the sand; beyond, a lip may win
    assert np.all(t.ground[dry] <= limit[dry])
    assert t.ground[land & (d > 40) & (d < 60)].max() > 40.0          # the banks still stand high further off


def test_sand_is_flat_and_zones_cover_the_tray(ground):
    g, t = ground
    assert np.all(t.ground[t.zone == SAND] == 0.0)
    assert np.isnan(t.water[t.zone == SAND]).all()
    assert (t.zone == WATER).sum() * 4.0 == pytest.approx(3.4e4, rel=0.1)     # ~3.5 dm2 of open water at 2 mm cells
