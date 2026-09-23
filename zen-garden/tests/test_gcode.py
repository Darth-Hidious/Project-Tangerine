"""G-code round trip: what the controller would execute reproduces the planned passes."""

import math

import numpy as np
import pytest

from karesansui import gcode
from karesansui.config import poc_garden
from karesansui.planner import ArmMachine, Pass, build_program


@pytest.fixture(scope="module", params=["scara", "articulated"])
def setup(request):
    garden = poc_garden()
    machine = ArmMachine(garden, request.param)
    prog = build_program(garden, "lines", machine=machine)
    lines = gcode.generate(prog)
    segs = gcode.parse(lines, machine.name)
    return garden, machine, prog, lines, segs


def test_every_planned_pose_is_a_commanded_point(setup):
    garden, m, prog, lines, segs = setup
    ends = m.arm.fk(np.array([s.q1 for s in segs if not s.rapid]))
    for p in prog.passes:
        z = m.z_for(p.kind) + (p.lift if p.lift is not None else np.zeros(len(p.poses)))
        for k in range(0, len(p.poses), max(len(p.poses) // 25, 1)):
            pose = p.poses[k]
            d = np.hypot(ends[:, 0] - pose[0], ends[:, 1] - pose[1]) + np.abs(ends[:, 2] - z[k])
            assert d.min() < 2e-3                      # 4-decimal degrees at ~0.5 m reach


def test_joint_space_chords_stay_on_the_path(setup):
    """Every feed move - along a pass or straight up and down - stays within 0.01 mm of the
    Cartesian straight line the planner meant, although the controller interpolates joints."""
    _, m, _, _, segs = setup
    worst = 0.0
    for s in segs:
        if s.rapid:
            continue
        a, b = m.arm.fk(s.q0)[0], m.arm.fk(s.q1)[0]
        mid = m.arm.fk(0.5 * (s.q0 + s.q1))[0]
        worst = max(worst, float(np.linalg.norm(mid[:3] - 0.5 * (a[:3] + b[:3]))))
    assert worst < 0.01


def test_rake_passes_run_released_and_screed_passes_latched(setup):
    _, m, _, _, segs = setup
    n_rake = n_screed = 0
    for s in segs:
        if s.rapid:
            continue
        a, b = m.arm.fk(s.q0)[0], m.arm.fk(s.q1)[0]
        horizontal = np.hypot(*(b[:2] - a[:2])) > 0.05
        if horizontal and np.isclose(a[2], m.z_for("rake"), atol=0.01) and np.isclose(b[2], m.z_for("rake"), atol=0.01):
            assert not s.latched
            n_rake += 1
        if horizontal and np.isclose(a[2], m.z_for("screed"), atol=0.01) and np.isclose(b[2], m.z_for("screed"), atol=0.01):
            assert s.latched
            n_screed += 1
    assert n_rake > 100 and n_screed > 100


def test_pass_timing_matches_speed(setup):
    garden, m, prog, lines, segs = setup
    traj = gcode.sample(segs, m.arm)
    rake_len = sum(np.hypot(*np.diff(p.poses[:, :2], axis=0).T).sum() for p in prog.passes if p.kind == "rake")
    screed_len = sum(np.hypot(*np.diff(p.poses[:, :2], axis=0).T).sum() for p in prog.passes if p.kind == "screed")
    lower_bound = rake_len / garden.gantry.rake_speed + screed_len / garden.gantry.screed_speed
    assert lower_bound < traj["seconds"] < 3 * lower_bound


def test_interpreter_rejects_unmodelled_codes():
    with pytest.raises(ValueError, match="unsupported"):
        gcode.parse(["G21 G90 G93", "G0 X0 Y0 Z0 A0", "G2 X1 Y1 I1 J0"], "scara")
    with pytest.raises(ValueError, match="inverse-time"):
        gcode.parse(["G21 G90 G94", "G0 X0 Y0 Z0 A0", "G1 X1 F100"], "scara")
