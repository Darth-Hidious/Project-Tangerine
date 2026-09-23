"""Planner checks: feasibility rules, collisions, groove geometry and travel routing."""

import dataclasses
import math

import numpy as np
import pytest
import shapely
from shapely.geometry import LineString, Point

from karesansui import default_garden
from karesansui.config import Stone
from karesansui.geometry import free_space, poses_valid, allowed_region
from karesansui.patterns import PatternError, waves
from karesansui.planner import Pass, Router, Travel, build_program

PATTERNS = ["lines", "waves", "ripples", "spiral"]


@pytest.fixture(scope="module")
def garden():
    return default_garden()


@pytest.fixture(scope="module")
def programs(garden):
    return {p: build_program(garden, p) for p in PATTERNS}


def test_original_wave_spec_is_rejected(garden):
    with pytest.raises(PatternError, match="inner tines would run backwards"):
        waves(garden, wavelength=60.0, amplitude=15.0)


@pytest.mark.parametrize("pattern", PATTERNS)
def test_patterns_plan_without_collisions(programs, pattern):
    r = programs[pattern].report
    assert r.ok, r.errors
    assert r.collisions == 0
    assert r.poses_checked > 10_000
    assert r.rake_mm > 5_000 and r.screed_mm > 10_000


@pytest.mark.parametrize("pattern", PATTERNS)
def test_no_rake_pass_is_tighter_than_the_hard_limit(programs, garden, pattern):
    assert programs[pattern].report.min_radius_mm >= garden.rake.min_radius_hard


def test_lane_and_ring_grooves_keep_the_pitch(programs, garden):
    spacing = programs["lines"].report.spacing
    for kind, q in spacing.items():
        assert q[1] == pytest.approx(garden.rake.pitch, abs=0.2), kind
        assert q[50] == pytest.approx(garden.rake.pitch, abs=0.2), kind


def test_wave_seams_match_closed_form(programs, garden):
    """Shifted sine lanes close up to lane_spacing * cos(max slope) - span at the steepest point."""
    amp, lam = 12.0, 400.0
    slope = math.atan(amp * 2 * math.pi / lam)
    expected = garden.rake.lane_spacing * math.cos(slope) - garden.rake.span
    q = programs["waves"].report.spacing["lane"]
    assert q[1] == pytest.approx(expected, abs=0.3)
    assert q[50] == pytest.approx(garden.rake.pitch, abs=0.2)


def test_frame_is_one_closed_pass(programs, garden):
    frames = [p for p in programs["ripples"].passes if p.label == "frame"]
    assert len(frames) == 1
    xy = frames[0].poses[:, :2]
    s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])
    # Loop length plus the closing overlap; the end sits on the path just past the start.
    start_to_end = LineString(xy[: np.searchsorted(s, garden.planner.closed_overlap + 5)]).distance(Point(xy[-1]))
    assert start_to_end < 1.0


@pytest.mark.parametrize("pattern", PATTERNS)
def test_heading_is_continuous_across_the_program(programs, pattern):
    steps = programs[pattern].steps
    for a, b in zip(steps[:-1], steps[1:]):
        assert np.allclose(a.poses[-1], b.poses[0], atol=1e-6)


@pytest.mark.parametrize("pattern", PATTERNS)
def test_head_only_turns_inside_free_space(programs, garden, pattern):
    free = free_space(garden)
    shapely.prepare(free)
    for step in programs[pattern].steps:
        if not isinstance(step, Travel):
            continue
        p = step.poses
        turning = np.flatnonzero(~np.isclose(np.diff(p[:, 2]), 0.0))
        for i in turning:
            assert free.buffer(1e-6).covers(LineString(p[i:i + 2, :2]))


def test_router_goes_around_the_triad(garden):
    free = free_space(garden)
    router = Router(free)
    a, b = np.array([480.0, 470.0]), np.array([1100.0, 470.0])
    assert free.covers(Point(a)) and free.covers(Point(b))
    path = router.route(a, b)
    length = np.hypot(*np.diff(path, axis=0).T).sum()
    assert length > np.hypot(*(b - a)) + 1.0
    assert all(free.covers(LineString(path[i:i + 2])) for i in range(len(path) - 1))


def test_screed_and_rake_passes_stay_in_allowed_region(programs, garden):
    region = allowed_region(garden)
    for pattern in PATTERNS:
        for p in programs[pattern].passes:
            assert poses_valid(p.poses, garden, region).all(), (pattern, p.label)


def test_stone_against_the_wall_is_planned_around():
    base = default_garden()
    wall_stone = Stone("wall stone", ((0, 360), (70, 350), (80, 400), (60, 450), (0, 440)), 60.0)
    garden = dataclasses.replace(base, stones=base.stones + (wall_stone,))
    prog = build_program(garden, "lines")
    assert prog.report.collisions == 0
    assert prog.report.ok
