"""Geometry checks against closed-form results (circles, sines, rectangles)."""

import numpy as np
import pytest
from shapely.geometry import Point

from karesansui import default_garden
from karesansui import geometry as g


def circle(r, n=4000, turns=1.0):
    t = np.linspace(0, 2 * np.pi * turns, n)
    return np.column_stack([r * np.cos(t), r * np.sin(t)])


def test_default_config_loads():
    garden = default_garden()
    assert garden.tray.width == 1200 and garden.tray.depth == 800
    assert len(garden.stones) == 4
    assert garden.rake.span == 100 and garden.rake.lane_spacing == 125
    assert garden.rake.tine_offsets == (-50, -25, 0, 25, 50)


def test_resample_preserves_length_and_spacing():
    xy = circle(200, n=50)
    out = g.resample(xy, 2.0)
    assert g.polyline_length(out) == pytest.approx(g.polyline_length(xy), rel=1e-3)
    seg = np.hypot(*np.diff(out, axis=0).T)
    assert seg.max() <= 2.0 + 1e-9 and seg.min() > 1.9


@pytest.mark.parametrize("radius", [60.0, 150.0, 400.0])
def test_curvature_of_exact_circle(radius):
    n = int(2 * np.pi * radius / 2.0)
    k = g.curvature(circle(radius, n=n))                          # points exactly on the circle
    assert np.allclose(k[1:-1], 1 / radius, rtol=1e-6)            # counter-clockwise = positive


@pytest.mark.parametrize("radius", [60.0, 150.0, 400.0])
def test_curvature_window_removes_polygonisation_noise(radius):
    xy = g.resample(circle(radius, n=4000), 2.0)                  # points on chords of a 4000-gon
    raw = g.curvature(xy)
    smooth = g.curvature(xy, half_window=10.0)
    assert np.allclose(smooth, 1 / radius, rtol=1e-3)
    assert np.abs(smooth * radius - 1).max() < np.abs(raw * radius - 1).max()


def test_curvature_window_still_catches_a_kink():
    xy = g.resample(np.array([[0.0, 0.0], [100.0, 0.0], [100.0 + 100 * np.cos(0.5), 100 * np.sin(0.5)]]), 2.0)
    assert g.min_radius(xy, half_window=10.0) < 52.5               # a 29 deg corner is not rakeable


def test_curvature_sign_clockwise():
    k = g.curvature(g.resample(circle(100)[::-1], 2.0))
    assert np.all(k[1:-1] < 0)


def test_sine_min_radius_matches_closed_form():
    lam, amp = 60.0, 15.0
    x = np.arange(0, 2 * lam, 0.01)
    xy = np.column_stack([x, amp * np.sin(2 * np.pi * x / lam)])
    expected = lam**2 / (4 * np.pi**2 * amp)                     # 1 / (A k^2) = 6.08 mm
    assert g.min_radius(xy) == pytest.approx(expected, rel=1e-3)


def test_inner_tines_reverse_on_original_wave_spec():
    """The original plan's lambda=60 mm, A=15 mm wave: 4 of 5 tines run backwards somewhere."""
    lam, amp = 60.0, 15.0
    x = np.arange(0, 2 * lam, 0.01)
    xy = np.column_stack([x, amp * np.sin(2 * np.pi * x / lam)])
    f = g.tine_speed_factors(xy, (-50, -25, 0, 25, 50))
    reverses = (f < 0).any(axis=1)
    assert reverses.tolist() == [True, True, False, True, True]


def test_tine_paths_are_parallel_on_a_circle():
    xy = g.resample(circle(300), 2.0)
    paths = g.tine_paths(xy, (-50, 0, 50))
    radii = np.hypot(paths[..., 0], paths[..., 1])
    assert np.allclose(radii[0], 350, atol=1e-2)   # left of a CCW circle is outside
    assert np.allclose(radii[2], 250, atol=1e-2)


def test_head_outline_and_swing_radius():
    garden = default_garden()
    local = g.head_outline_local(garden.rake, garden.screed)
    # Bar 12 x 111 mm on the axis, a 30 mm wide skid from 10 to 50 mm ahead, and the
    # (conservative) trapezoid joining them.
    poly = g.shapely.Polygon(local)
    assert poly.is_valid
    expected = 12 * 111 + (111 + 30) / 2 * 4 + 40 * 30
    assert poly.area == pytest.approx(expected)
    # The bar's corners, not the skid, set the swing radius once the skid is narrow.
    assert g.swing_radius(garden.rake, garden.screed) == pytest.approx(np.hypot(6, 55.5))


def test_footprint_is_rotation_invariant_in_area():
    garden = default_garden()
    poses = np.array([[600, 400, a] for a in np.linspace(0, 2 * np.pi, 7)])
    areas = g.shapely.area(g.head_footprints(poses, garden))
    assert np.allclose(areas, areas[0])


def test_pose_validity_near_walls_and_stones():
    garden = default_garden()
    poses = np.array([
        [150.0, 650.0, 0.0],     # open gravel, north-west quarter
        [150.0, 40.0, 0.0],      # bar would cross the south wall clearance
        [820.0, 400.0, 0.0],     # on top of the triad
        [1170.0, 400.0, 0.0],    # skid runs into the east wall
    ])
    assert g.poses_valid(poses, garden).tolist() == [True, False, False, False]


def test_free_space_excludes_swing_circle_near_obstacles():
    garden = default_garden()
    free = g.free_space(garden)
    r = g.swing_radius(garden.rake, garden.screed)
    assert free.contains(Point(150, 650))
    assert not free.contains(Point(40, 400))                      # closer than r to the west wall
    stones = g.stone_union(garden)
    for pt in [Point(xy) for xy in free.exterior.coords[::7]]:
        assert pt.distance(stones) >= garden.planner.stone_clearance + r - 1e-6


def test_split_runs():
    assert g.split_runs(np.array([0, 1, 1, 0, 1, 0, 0, 1], bool)) == [(1, 3), (4, 5), (7, 8)]
