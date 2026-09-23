"""Arm kinematics and statics against closed-form results."""

import math

import numpy as np
import pytest

from karesansui.arm import Articulated, Drive, Scara, line_wobble, make_arm
from karesansui.config import poc_garden


@pytest.fixture(scope="module")
def garden():
    """The proof-of-concept arm with its joints free over their full mechanical range: these tests
    check the kinematics everywhere, not the narrow limits the garden sets (tested in test_planner)."""
    import dataclasses
    g = poc_garden()
    from karesansui.config import ArticulatedCfg, ScaraCfg
    free_s = dataclasses.replace(g.arm.scara, j1_limits=ScaraCfg.j1_limits, j2_limits=ScaraCfg.j2_limits)
    free_a = dataclasses.replace(g.arm.articulated, j1_limits=ArticulatedCfg.j1_limits,
                                 j2_limits=ArticulatedCfg.j2_limits, j3_limits=ArticulatedCfg.j3_limits)
    return dataclasses.replace(g, arm=dataclasses.replace(g.arm, scara=free_s, articulated=free_a))


def random_poses(arm_base, n, rmin, rmax, z, seed=0):
    rng = np.random.default_rng(seed)
    r = rng.uniform(rmin, rmax, n)
    a = rng.uniform(math.radians(100), math.radians(260), n)       # towards the sand, left of the base
    heading = rng.uniform(-math.pi, math.pi, n)
    return np.column_stack([arm_base[0] + r * np.cos(a), arm_base[1] + r * np.sin(a), np.full(n, z), heading])


@pytest.mark.parametrize("elbow", [1, -1])
def test_scara_ik_inverts_fk(garden, elbow):
    arm = Scara.from_config(garden.arm, elbow=elbow)
    poses = random_poses(arm.base, 400, 130, 450, 12.0)
    q, ok = arm.ik(poses)
    assert ok.all()
    back = arm.fk(q)
    assert np.allclose(back[:, :3], poses[:, :3], atol=1e-9)
    assert np.allclose(np.cos(back[:, 3] - poses[:, 3]), 1.0)


def test_scara_reach_limits(garden):
    arm = Scara.from_config(garden.arm)
    too_far = np.array([[arm.base[0] - 465.0, arm.base[1], 0.0, 0.0]])
    fold = math.radians(garden.arm.scara.j2_limits[1])
    r_min = math.sqrt(arm.l1**2 + arm.l2**2 + 2 * arm.l1 * arm.l2 * math.cos(fold))
    too_close = np.array([[arm.base[0] - (r_min - 2.0), arm.base[1], 0.0, 0.0]])
    just_ok = np.array([[arm.base[0] - (r_min + 2.0), arm.base[1], 0.0, 0.0]])
    assert not arm.ik(too_far)[1][0]
    assert not arm.ik(too_close)[1][0]
    assert arm.ik(just_ok)[1][0]


@pytest.mark.parametrize("kind", ["scara", "articulated"])
def test_joint_limit_check_agrees_with_ik(garden, kind):
    """in_limits (used to check G-code) must accept exactly what the IK calls reachable."""
    arm = make_arm(garden, kind)
    poses = random_poses(arm.base, 2000, 60, 560, 12.0, seed=3)
    q, ok = arm.ik(poses)
    reach = arm.l1 + arm.l2 if kind == "scara" else arm.lu + arm.lf
    inside = np.hypot(*(poses[:, :2] - arm.base).T) < reach - 60     # both sides of every limit get sampled
    assert ok.any() and (~ok & inside).any()
    assert np.array_equal(arm.in_limits(q)[inside], ok[inside])


def test_scara_lift_stroke_limits(garden):
    arm = make_arm(garden, "scara")
    q, ok = arm.ik(random_poses(arm.base, 1, 250, 250, 0.0))
    assert ok[0] and arm.in_limits(q)[0]
    for z, expected in ((garden.arm.scara.lift, True), (garden.arm.scara.lift + 0.5, False), (-0.5, False)):
        q[0, 2] = z
        assert arm.in_limits(q)[0] == expected


@pytest.mark.parametrize("kind", ["scara", "articulated"])
def test_jacobian_matches_finite_differences(garden, kind):
    arm = make_arm(garden, kind)
    poses = random_poses(arm.base, 50, 180, 420, 0.0, seed=3)
    q, ok = arm.ik(poses)
    q = q[ok]
    J = arm.jacobian_xy(q)
    eps = 1e-6
    for k in range(arm.n_joints):
        dq = np.zeros(arm.n_joints)
        dq[k] = eps
        fd = (arm.fk(q + dq)[:, :2] - arm.fk(q - dq)[:, :2]) / (2 * eps)
        assert np.allclose(J[:, :, k], fd, atol=1e-5)


@pytest.mark.parametrize("z", [0.0, 40.0, 85.0])
def test_articulated_ik_inverts_fk_with_a_vertical_tool(garden, z):
    arm = Articulated.from_config(garden.arm)
    poses = random_poses(arm.base, 300, 150, 420, z, seed=1)
    q, ok = arm.ik(poses)
    assert ok.mean() > 0.95
    back = arm.fk(q[ok])
    assert np.allclose(back[:, :3], poses[ok, :3], atol=1e-9)
    tilt = q[ok, 1] + q[ok, 2] + q[ok, 3] + math.pi / 2
    assert np.allclose(tilt, 0.0)


def test_articulated_holding_torque_when_stretched_flat(garden):
    arm = Articulated.from_config(garden.arm)
    q = np.array([[0.0, 0.0, 0.0, -math.pi / 2, 0.0]])
    tau = arm.gravity_torques(q)[0]
    lu, lf = arm.lu / 1000, arm.lf / 1000
    shoulder = 9.81 * (arm.mu * lu / 2 + arm.mf * (lu + lf / 2) + arm.mt * (lu + lf))
    elbow = 9.81 * (arm.mf * lf / 2 + arm.mt * lf)
    assert tau[1] == pytest.approx(shoulder)
    assert tau[2] == pytest.approx(elbow)
    assert tau[3] == pytest.approx(0.0)


def test_scara_motors_hold_no_weight(garden):
    arm = Scara.from_config(garden.arm)
    q, _ = arm.ik(random_poses(arm.base, 20, 150, 400, 0.0))
    tau = arm.gravity_torques(q)
    assert np.allclose(tau[:, [0, 1, 3]], 0.0)
    assert np.allclose(tau[:, 2], arm.mt * 9.81)


def test_tool_yaw_stays_continuous_through_a_full_turn(garden):
    arm = Scara.from_config(garden.arm)
    headings = np.linspace(0, 4 * math.pi, 200)
    poses = np.column_stack([np.full(200, 400.0), np.full(200, 225.0), np.zeros(200), headings])
    q, ok = arm.ik(poses)
    assert ok.all()
    assert np.abs(np.diff(q[:, 3])).max() < 0.1


def test_line_wobble_matches_hand_calculation(garden):
    """SCARA stretched straight out from its base and raking radially: base and elbow errors
    push the tool around the base, i.e. straight across a radial groove."""
    arm = Scara.from_config(garden.arm)
    q = np.array([[0.0, 0.0, 0.0, 0.0]])                             # stretched along zero_deg
    heading = arm.fk(q)[:, 3]
    assert math.cos(heading[0] - math.radians(garden.arm.zero_deg)) == pytest.approx(1.0)
    drive = Drive("test", backlash=0.2, resolution=0.0)
    centre, _ = line_wobble(arm, q, heading, drive, garden.rake.tine_offsets)
    e = math.radians(0.1)                                            # half the dead band
    assert centre[0] == pytest.approx((arm.l1 + arm.l2) * e + arm.l2 * e, rel=1e-9)
    worse, _ = line_wobble(arm, q, heading, Drive("x2", 0.4, 0.0), garden.rake.tine_offsets)
    assert worse[0] == pytest.approx(2 * centre[0])
    # The same pose raking a circle around the base: the joint errors run along the groove.
    tangential, _ = line_wobble(arm, q, heading + math.pi / 2, drive, garden.rake.tine_offsets)
    assert tangential[0] == pytest.approx(0.0, abs=1e-9)
