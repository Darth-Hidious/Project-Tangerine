"""Sand-bed physics checks: conservation laws, slope limits and depth control.

A small stone-free tray keeps these fast; the kernels are the same as for the full garden.
"""

import dataclasses
import math

import numpy as np
import pytest

from karesansui import default_garden
from karesansui.config import Tray
from karesansui.sim import Bed, unevenness_field


@pytest.fixture(scope="module")
def garden():
    g = default_garden()
    return dataclasses.replace(g, stones=(), tray=Tray(width=300.0, depth=200.0, bed_depth=35.0))


def stroke(bed, garden, y=100.0, x0=40.0, x1=260.0, rigid=False):
    x = np.arange(x0, x1, 0.5)
    ys = np.full_like(x, y)
    z = garden.tray.bed_depth + (garden.rake.hang if rigid else garden.gantry.rake_clearance)
    return bed.rake(x, ys, np.zeros_like(x), np.full_like(x, z), rigid=rigid)


def test_raking_conserves_mass_and_respects_repose(garden):
    bed = Bed(garden)
    v0 = bed.volume()
    st = stroke(bed, garden)
    assert st.moved_mm3 > 0
    assert bed.volume() == pytest.approx(v0, rel=1e-12)
    assert bed.max_slope_excess() <= bed.cfg.relax_tol + 1e-9


def test_grooves_sit_under_the_tines_with_symmetric_ridges(garden):
    bed = Bed(garden)
    stroke(bed, garden, y=100.0)
    prof = bed.h[:, 150]
    y = (np.arange(len(prof)) + 0.5) * bed.cfg.dx
    for d in garden.rake.tine_offsets:
        near = np.abs(y - (100.0 + d)) <= 6.0
        assert abs(y[near][np.argmin(prof[near])] - (100.0 + d)) <= 1.0
    left = prof[(y > 88) & (y < 100)].max()
    right = prof[(y > 100) & (y < 112)].max()
    assert left == pytest.approx(right, abs=0.3)
    assert prof.min() < garden.tray.bed_depth - 1.0 and prof.max() > garden.tray.bed_depth + 1.0


def test_pile_relaxes_to_the_angle_of_repose(garden):
    bed = Bed(garden)
    v0 = bed.volume()
    bed.h[95:105, 145:155] += 30.0                   # a 10 x 10 x 30 mm column of grit
    v1 = bed.volume()
    bed.relax()
    assert bed.volume() == pytest.approx(v1, rel=1e-12)
    assert v1 - v0 == pytest.approx(3000.0)
    assert bed.max_slope_excess() <= bed.cfg.relax_tol + 1e-9
    # The cone's flank along the x axis stands at the repose angle.
    row = bed.h[100, :] - garden.tray.bed_depth
    flank = row[152:170]
    steep = -np.diff(flank)[np.diff(flank) < -1e-3]
    assert steep.max() == pytest.approx(math.tan(math.radians(garden.grit.repose_deg)), rel=0.05)


def test_floating_skid_holds_groove_depth_on_a_tilted_bed_and_rigid_head_does_not(garden):
    """The bed rises 4.8 mm along the measured stretch. A floating head's grooves must not
    follow that rise; a rigid head's must (it cuts deeper as the bed climbs towards it)."""
    ny, nx = int(garden.tray.depth), int(garden.tray.width)
    tilt = np.broadcast_to(((np.arange(nx) + 0.5) - nx / 2) * 0.03, (ny, nx)).copy()   # 1.7 deg
    cols = np.arange(70, 231, 10)
    rise = {}
    for rigid in (False, True):
        bed = Bed(garden, unevenness=tilt)
        stroke(bed, garden, y=100.0, rigid=rigid)
        base = garden.tray.bed_depth + tilt[100, cols]
        depth = base - bed.h[95:106, cols].min(axis=0)
        rise[rigid] = np.polyfit(cols, depth, 1)[0] * (cols[-1] - cols[0])
    assert abs(rise[False]) < 0.3
    assert rise[True] > 2.0


@pytest.mark.parametrize("direction", ["along", "across"])
def test_screed_erases_a_raked_field_without_losing_grit(garden, direction):
    bed = Bed(garden)
    for y in (60.0, 140.0):
        stroke(bed, garden, y=y)
    v0 = bed.volume()
    region = (slice(45, 155), slice(70, 230))
    rough = bed.h[region].std()
    z = garden.tray.bed_depth
    # Each lane drops its leftover berm where it ends, so lanes end outside the measured area
    # (the planner does the same: screed lanes finish in the border the pattern never uses).
    if direction == "along":                               # lanes parallel to the grooves
        for y in np.arange(45.0, 160.0, 25.0):
            x = np.arange(289.0, 12.0, -0.5)
            bed.screed(x, np.full_like(x, y), np.full_like(x, math.pi), np.full_like(x, z))
    else:                                                  # lanes across the grooves
        for x0 in np.arange(60.0, 260.0, 25.0):
            y = np.arange(189.0, 12.0, -0.5)
            bed.screed(np.full_like(y, x0), y, np.full_like(y, -math.pi / 2), np.full_like(y, z))
    assert bed.volume() == pytest.approx(v0, rel=1e-12)
    assert bed.lost_mm3 == 0.0
    assert bed.h[region].std() < 0.25 * rough


def test_unevenness_field_is_zero_mean_with_requested_peak(garden):
    f = unevenness_field(garden, amplitude=3.0, correlation=40.0, seed=7)
    assert f.shape == (200, 300)
    assert abs(f.mean()) < 1e-9
    assert np.abs(f).max() == pytest.approx(3.0)
