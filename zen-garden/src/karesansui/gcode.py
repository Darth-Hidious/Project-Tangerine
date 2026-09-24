"""Joint-space G-code for the arm, and an interpreter that turns it back into tool poses.

The planner works in Cartesian poses; the controller (FluidNC on an ESP32, which supports
up to six axes, G93 inverse-time feed and synchronised M67 analog outputs) moves joints.
Inverse kinematics runs here, on the host, and every axis letter is a joint:

    SCARA        X = q1 base (deg)   Y = q2 elbow (deg)   Z = lift (mm)      A = q4 tool yaw (deg)
    articulated  X = q1 base (deg)   Y = q2 shoulder      Z = q3 elbow       A = q4 wrist pitch   B = q5 tool yaw

The controller interpolates each G1 linearly in joint space. Consecutive points are ~1 mm
apart along the path, so the tool's deviation from the straight Cartesian chord is
microscopic (checked by tests).

Feed is G93 inverse time: F = 60 / seconds for that move, so timing is exact whatever the
axis units. The rake latch is a hobby servo on analog output 0 at 50 Hz: M67 E0 Q10 latches
the head up (2.0 ms pulse), Q5 releases it to float (1.0 ms). M67 waits for queued motion,
so the latch changes exactly between moves.

The interpreter implements exactly this subset (G0 G1 G4 G21 G90 G93 G94 M67 M2 M30 and
comments) and rejects anything else, so a generated file cannot silently mean something
the simulation did not model.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

from .geometry import cumulative_length
from .planner import Pass, Program, Travel

LATCHED, RELEASED = 10.0, 5.0
AXES = {"scara": "XYZA", "articulated": "XYZAB"}
ANGLE_AXES = {"scara": (0, 1, 3), "articulated": (0, 1, 2, 3, 4)}


@dataclass(frozen=True)
class Limits:
    joint_speed: float = 60.0      # deg/s, rotary joints, rapids
    lift_speed: float = 20.0       # mm/s, SCARA lift
    settle: float = 0.3            # s, dwell after lowering or latching the head


def _fmt(values: np.ndarray, letters: str) -> str:
    return " ".join(f"{a}{v:.4f}" for a, v in zip(letters, values))


def to_axes(q: np.ndarray, kind: str) -> np.ndarray:
    """Joint values (rad, mm) -> axis values (deg, mm)."""
    v = np.array(q, float, copy=True)
    idx = list(ANGLE_AXES[kind])
    v[..., idx] = np.degrees(v[..., idx])
    return v


def from_axes(v: np.ndarray, kind: str) -> np.ndarray:
    q = np.array(v, float, copy=True)
    idx = list(ANGLE_AXES[kind])
    q[..., idx] = np.radians(q[..., idx])
    return q


def generate(program: Program, limits: Limits = Limits()) -> list[str]:
    """G-code for an arm program. Every pose is converted with the machine's own IK."""
    m = program.machine
    kind, letters = m.name, AXES[m.name]
    speeds = {"rake": program.garden.gantry.rake_speed, "screed": program.garden.gantry.screed_speed}
    out = [
        f"; karesansui {program.pattern} program for a {kind} arm",
        f"; axes: {', '.join(f'{a}={n}' for a, n in zip(letters, m.arm.joint_names))}",
        "G21 G90 G93",
        f"M67 E0 Q{LATCHED:g}",
    ]
    q_prev = None
    latched = True

    def rapid(qs):
        nonlocal q_prev
        for q in qs:
            out.append("G0 " + _fmt(to_axes(q, kind), letters))
            q_prev = q

    def feed(q, seconds):
        nonlocal q_prev
        out.append("G1 " + _fmt(to_axes(q, kind), letters) + f" F{60.0 / max(seconds, 1e-3):.4f}")
        q_prev = q

    def vertical(z_to: float, step_mm: float = 2.0):
        """Straight up or down from the current pose, in small Cartesian steps: a single joint-space
        move would swing an articulated arm's tool sideways by millimetres on the way."""
        pose = m.arm.fk(q_prev)[0]
        n = max(int(math.ceil(abs(z_to - pose[2]) / step_mm)), 1)
        for zz in np.linspace(pose[2], z_to, n + 1)[1:]:
            q, ok = m.ik(np.array([[pose[0], pose[1], pose[3]]]), zz, q_prev=q_prev)
            if not ok[0]:
                raise ValueError("vertical move leaves the arm's reach")
            feed(q[0], abs(z_to - pose[2]) / n / limits.lift_speed)

    for step in program.steps:
        if isinstance(step, Travel):
            j = step.joints
            # First and last waypoints are the vertical moves off and onto the passes.
            if q_prev is not None:
                vertical(float(step.z[1]))
                if not latched:
                    out.append(f"M67 E0 Q{LATCHED:g}")
                    out.append(f"G4 P{limits.settle:g}")
                    latched = True
            rapid(j[1:-1])
            continue
        p: Pass = step
        z = m.z_for(p.kind) + (p.lift if p.lift is not None else 0.0)
        q, ok = m.ik(p.poses, z, q_prev=q_prev)
        if not ok.all():
            raise ValueError(f"{p.label}: pose out of reach while generating G-code")
        out.append(f"; {p.kind} pass {p.label}")
        want_latched = p.kind != "rake"
        if want_latched != latched:
            out.append(f"M67 E0 Q{LATCHED if want_latched else RELEASED:g}")
            latched = want_latched
        if q_prev is None:
            feed(q[0], 1.0)
        else:
            vertical(float(np.broadcast_to(z, (len(q),))[0]))
            feed(q[0], 0.05)                                 # snap onto the exact pass start
        out.append(f"G4 P{limits.settle:g}")
        seg = np.hypot(*np.diff(p.poses[:, :2], axis=0).T)
        dth = np.abs(np.diff(p.poses[:, 2]))
        v = speeds[p.kind]
        for k in range(1, len(q)):
            t = max(seg[k - 1] / v, math.degrees(dth[k - 1]) / limits.joint_speed)
            feed(q[k], t)
    # Park: the last travel step already ended at the park pose; lift there and latch.
    out.append(f"M67 E0 Q{LATCHED:g}")
    out.append("M2")
    return out


# ============================================================================ interpreter
@dataclass
class Segment:
    q0: np.ndarray
    q1: np.ndarray
    seconds: float
    rapid: bool
    latched: bool


_WORD = re.compile(r"([A-Z])\s*(-?\d+(?:\.\d*)?|-?\.\d+)")
_OK_G = {0, 1, 4, 21, 90, 93, 94}


def parse(lines: list[str], kind: str, limits: Limits = Limits()) -> list[Segment]:
    letters = AXES[kind]
    pos = None
    latched = True
    inverse_time = False
    segs: list[Segment] = []
    for n, raw in enumerate(lines, 1):
        line = raw.split(";", 1)[0]
        line = re.sub(r"\(.*?\)", "", line).strip().upper()
        if not line:
            continue
        words = _WORD.findall(line)
        gs = [int(float(v)) for k, v in words if k == "G"]
        ms = [int(float(v)) for k, v in words if k == "M"]
        vals = {k: float(v) for k, v in words if k not in "GM"}
        bad = [g for g in gs if g not in _OK_G] + [mm for mm in ms if mm not in (2, 30, 67)]
        if bad:
            raise ValueError(f"line {n}: unsupported code(s) {bad}: {raw!r}")
        if 93 in gs:
            inverse_time = True
        if 94 in gs:
            inverse_time = False
        if 67 in ms:
            latched = vals.get("Q", LATCHED) >= (LATCHED + RELEASED) / 2
            continue
        if 4 in gs:
            if pos is not None:                              # a dwell is a zero-length move in time
                q = from_axes(pos, kind)
                segs.append(Segment(q, q, vals.get("P", 0.0), False, latched))
            continue
        if not any(k in letters for k in vals):
            continue
        target = np.array([vals.get(a, np.nan) for a in letters])
        if pos is None:
            if np.isnan(target).any():
                raise ValueError(f"line {n}: first move must set every axis")
            pos = target
            continue
        target = np.where(np.isnan(target), pos, target)
        rapid = 0 in gs
        if rapid:
            span = np.abs(target - pos)
            rot = span[list(ANGLE_AXES[kind])].max(initial=0.0) / limits.joint_speed
            lin = 0.0 if kind != "scara" else span[2] / limits.lift_speed
            seconds = max(rot, lin, 1e-3)
        else:
            if not inverse_time or "F" not in vals:
                raise ValueError(f"line {n}: G1 needs G93 inverse-time feed and an F word")
            seconds = 60.0 / vals["F"]
        segs.append(Segment(from_axes(pos, kind), from_axes(target, kind), seconds, rapid, latched))
        pos = target
    return segs


def sample(segs: list[Segment], arm, step_mm: float = 0.5) -> dict:
    """Densify the parsed motion (linear in joint space, as the controller does) into tool poses."""
    xs, ys, zs, hs, rel, ts = [], [], [], [], [], []
    t = 0.0
    for s in segs:
        a, b = arm.fk(s.q0)[0], arm.fk(s.q1)[0]
        n = max(int(math.ceil(max(np.hypot(*(b[:2] - a[:2])), abs(b[2] - a[2])) / step_mm)), 1)
        f = np.linspace(0, 1, n + 1)[1:, None]
        qs = s.q0 + f * (s.q1 - s.q0)
        poses = arm.fk(qs)
        xs.append(poses[:, 0]); ys.append(poses[:, 1]); zs.append(poses[:, 2]); hs.append(poses[:, 3])
        rel.append(np.full(n, not s.latched))
        ts.append(t + f[:, 0] * s.seconds)
        t += s.seconds
    cat = lambda v: np.concatenate(v) if v else np.zeros(0)
    return {"x": cat(xs), "y": cat(ys), "z": cat(zs), "heading": cat(hs), "released": cat(rel).astype(bool),
            "t": cat(ts), "seconds": t}


def run_on_bed(bed, traj: dict) -> None:
    """Apply a sampled trajectory to a sand bed: rake wherever the head is released, screed
    wherever the latched blade is lower than the rake's working height (at the sand, or rising
    off it at the end of a pass); the arm's z is the blade edge above the sand surface."""
    s = bed.garden.tray.bed_depth
    z_abs = traj["z"] + s
    released = traj["released"]
    in_sand = traj["z"] < bed.garden.gantry.rake_clearance   # blade on or just above the sand
    # Split into runs of constant tool state; travel samples above the surface are skipped by the kernels.
    state = np.where(released, 2, np.where(in_sand, 1, 0))
    edges = np.flatnonzero(np.diff(state)) + 1
    for a, b in zip(np.concatenate([[0], edges]), np.concatenate([edges, [len(state)]])):
        sl = slice(a, b)
        if state[a] == 2:
            bed.rake(traj["x"][sl], traj["y"][sl], traj["heading"][sl], z_abs[sl])
        elif state[a] == 1:
            bed.screed(traj["x"][sl], traj["y"][sl], traj["heading"][sl], z_abs[sl])
