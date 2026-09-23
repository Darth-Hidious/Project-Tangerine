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


# --------------------------------------------------------------------------- proof of concept (arm)
from karesansui.config import poc_garden
from karesansui.planner import ArmMachine


@pytest.fixture(scope="module")
def poc():
    return poc_garden()


@pytest.mark.parametrize("kind", ["scara", "articulated"])
@pytest.mark.parametrize("pattern", PATTERNS)
def test_poc_patterns_plan_for_both_arms(poc, kind, pattern):
    prog = build_program(poc, pattern, machine=ArmMachine(poc, kind))
    r = prog.report
    assert r.ok, r.errors
    assert r.collisions == 0 and r.poses_checked > 2_000
    assert r.coverage > 0.8
    assert r.min_radius_mm >= poc.rake.min_radius_hard


def test_poc_travel_lifts_over_everything_in_reach(poc):
    machine = ArmMachine(poc, "scara")
    assert machine.obstacle_clearance() == []
    prog = build_program(poc, "lines", machine=machine)
    for step in prog.steps:
        if isinstance(step, Travel):
            assert step.z[0] <= step.z.max() and step.z[1:-1].min() == pytest.approx(machine.z_travel)
            # Every travel waypoint is a reachable arm configuration that reproduces the pose.
            back = machine.arm.fk(step.joints)
            assert np.allclose(back[:, :2], step.poses[:, :2], atol=1e-6)


def test_obstacle_taller_than_the_lift_is_reported(poc):
    lanterns = tuple(dataclasses.replace(l, height=150.0) if l.name == "front-right lantern" else l
                     for l in poc.lanterns)
    msgs = ArmMachine(dataclasses.replace(poc, lanterns=lanterns), "scara").obstacle_clearance()
    assert len(msgs) == 1 and "front-right lantern" in msgs[0]            # below the links: head only


def test_bonsai_is_clear_only_while_it_stays_out_of_reach(poc):
    """The tree is taller than the travel lift, so it must sit beyond the arm's reach plus the
    head's size; the same tree 100 mm closer to the arm is reported."""
    tree = next(f for f in poc.features if f.kind == "tree")
    assert tree.height > poc.arm.travel_lift
    assert ArmMachine(poc, "scara").obstacle_clearance() == []
    moved = dataclasses.replace(tree, outline=tuple((x + 100.0, y) for x, y in tree.outline))
    garden = dataclasses.replace(poc, features=tuple(moved if f is tree else f for f in poc.features))
    msgs = ArmMachine(garden, "scara").obstacle_clearance()
    assert len(msgs) == 2 and all("bonsai" in m for m in msgs)            # taller than the links: both


@pytest.mark.parametrize("kind", ["scara", "articulated"])
def test_erase_reaches_all_but_the_stone_gaps_and_the_edge_strip(poc, kind):
    """The screed loop along the sand's edge plus lanes and island rings pass the blade over at
    least 90 % of the open sand; the rest is the gap between the paired stones, the clearance
    band around them and the strip the head keeps from the edge."""
    prog = build_program(poc, "ripples", machine=ArmMachine(poc, kind))
    assert prog.report.erase_coverage > 0.9
    assert any(p.label == "screed:frame" for p in prog.passes)


def test_joint_limits_keep_the_arm_off_the_tree_corner(poc):
    """The sweep check over the joint-limit box (the limits are also hard stops): the tree clears
    the head and the links by more than the margin; with the joints free over their mechanical
    range the same tree is inside the arm's sweep."""
    tree = next(f for f in poc.features if f.kind == "tree")
    from shapely.geometry import Polygon
    head, links = ArmMachine(poc, "scara").sweep_clearance(Polygon(tree.outline))
    assert head > 10.0 and links > 10.0
    free = dataclasses.replace(poc.arm.scara, j1_limits=(-170.0, 170.0), j2_limits=(-150.0, 150.0))
    head_w, links_w = ArmMachine(dataclasses.replace(poc, arm=dataclasses.replace(poc.arm, scara=free)),
                                 "scara").sweep_clearance(Polygon(tree.outline))
    assert min(head_w, links_w) < 0.0
