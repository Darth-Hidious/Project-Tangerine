"""The ground around the sand: every bank holds its water, and the hill stays inside its walls."""

import numpy as np
import pytest
from scipy.ndimage import binary_dilation, maximum_filter

from karesansui.config import poc_garden
from karesansui.landscape import BANK, KERB, LIVING, PRESERVED, SAND, WATER, terrain


@pytest.fixture(scope="module")
def ground():
    g = poc_garden()
    return g, terrain(g, dx=2.0, overlays=False)


def test_no_water_stands_above_its_banks(ground):
    g, t = ground
    wet = ~np.isnan(t.water)
    bank = binary_dilation(wet) & ~wet & np.isin(t.zone, (LIVING, PRESERVED, KERB))
    highest_neighbour = maximum_filter(np.where(wet, t.water, -np.inf), size=3)
    assert bank.sum() > 300
    assert np.all(t.ground[bank] >= highest_neighbour[bank] + BANK - 3.0)     # 3 mm of moss-noise slack
    assert np.all(t.ground[wet] < t.water[wet])                               # the bed is under the water


def test_the_ground_stays_below_the_tray_walls(ground):
    g, t = ground
    X, Y = np.meshgrid(t.x, t.y)
    land = np.isin(t.zone, (LIVING, PRESERVED, KERB))
    near = lambda d: (d <= 10.0) & land
    back_left = near(np.minimum(X, g.tray.depth - Y))
    front_right = near(np.minimum(g.tray.width - X, Y))
    assert t.ground[back_left].max() < g.tray.back_wall_height - g.tray.bed_depth
    assert t.ground[front_right].max() < g.tray.wall_height - g.tray.bed_depth


def test_sand_is_flat_and_zones_cover_the_tray(ground):
    g, t = ground
    assert np.all(t.ground[t.zone == SAND] == 0.0)
    assert np.isnan(t.water[t.zone == SAND]).all()
    assert (t.zone == WATER).sum() * 4.0 == pytest.approx(3.4e4, rel=0.1)     # ~3.5 dm2 of open water at 2 mm cells
