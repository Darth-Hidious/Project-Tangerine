"""Arm models for raking: SCARA and a 5-joint articulated arm.

Both carry the same floating rake head. The *tool point* is the head's yaw axis at the
skid sole; a tool pose is (x, y, z_sole, heading) with z measured from the sand surface.
The head floats vertically on its own slide, so the arm only has to put the carriage inside
the float range: vertical arm errors are absorbed, horizontal ones draw wobbly lines.

SCARA (joints: base yaw q1, elbow yaw q2, vertical lift z, tool yaw q4)
    Links move in a horizontal plane above everything in the garden; gravity loads only
    bearings, not motors.
Articulated (joints: base yaw q1, shoulder pitch q2, elbow pitch q3, wrist pitch q4,
tool yaw q5)
    Looks like the render: an arm reaching over the sand. Shoulder and elbow hold the arm's
    weight all the time; the wrist pitch keeps the tool vertical.

Angles are radians internally and degrees in configs and G-code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import Arm, Garden

G = 9.81


class Unreachable(ValueError):
    pass


# ============================================================================ SCARA
@dataclass
class Scara:
    base: np.ndarray            # (2,)
    zero: float                 # tray-frame direction of q1 = 0 (rad)
    l1: float
    l2: float
    link_height: float
    lift: float
    lim1: tuple[float, float]   # rad
    lim2: tuple[float, float]
    m1: float
    m2: float
    mt: float
    elbow: int = 1              # +1: elbow on the left of the base->tool line (seen from above)
    n_joints: int = 4
    joint_names = ("q1 base", "q2 elbow", "z lift", "q4 tool yaw")

    @classmethod
    def from_config(cls, arm: Arm, elbow: int = 1) -> "Scara":
        c = arm.scara
        return cls(np.asarray(arm.base, float), math.radians(arm.zero_deg), c.link1, c.link2, c.link_height, c.lift,
                   tuple(np.radians(c.j1_limits)), tuple(np.radians(c.j2_limits)),
                   c.link1_mass, c.link2_mass, c.tool_mass, elbow)

    # ---- kinematics
    def fk(self, q: np.ndarray) -> np.ndarray:
        """Joints (N, 4) -> tool poses (N, 4): x, y, z, heading."""
        q = np.atleast_2d(q)
        a1 = q[:, 0] + self.zero
        a12 = a1 + q[:, 1]
        x = self.base[0] + self.l1 * np.cos(a1) + self.l2 * np.cos(a12)
        y = self.base[1] + self.l1 * np.sin(a1) + self.l2 * np.sin(a12)
        return np.column_stack([x, y, q[:, 2], a12 + q[:, 3]])

    def elbows(self, q: np.ndarray) -> np.ndarray:
        q = np.atleast_2d(q)
        a1 = q[:, 0] + self.zero
        return self.base + self.l1 * np.column_stack([np.cos(a1), np.sin(a1)])

    def ik(self, poses: np.ndarray, q_prev: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Tool poses (N, 4) -> joints (N, 4) and a reachability mask (joint limits included).

        Headings stay continuous (the tool yaw joint unwraps along the sequence)."""
        poses = np.atleast_2d(poses)
        d = poses[:, :2] - self.base
        r2 = (d**2).sum(axis=1)
        c2 = (r2 - self.l1**2 - self.l2**2) / (2 * self.l1 * self.l2)
        ok = np.abs(c2) <= 1.0
        q2 = self.elbow * np.arccos(np.clip(c2, -1, 1))
        a1 = np.arctan2(d[:, 1], d[:, 0]) - np.arctan2(self.l2 * np.sin(q2), self.l1 + self.l2 * np.cos(q2))
        q1 = _wrap(a1 - self.zero)
        q4 = poses[:, 3] - (q1 + self.zero) - q2
        q4 = _continuous(q4, None if q_prev is None else q_prev[3])
        ok &= _within(q1, self.lim1) & _within(q2, self.lim2)
        return np.column_stack([q1, q2, poses[:, 2], q4]), ok

    def in_limits(self, q: np.ndarray) -> np.ndarray:
        """Joint limits per configuration. The lift strokes from z = 0 (latched blade edge on the
        sand, as when screeding) up to ``lift``; the tool yaw turns without limit (slip ring)."""
        q = np.atleast_2d(q)
        return _within(q[:, 0], self.lim1) & _within(q[:, 1], self.lim2) & _within(q[:, 2], (0.0, self.lift))

    def jacobian_xy(self, q: np.ndarray) -> np.ndarray:
        """(N, 2, n_joints) sensitivity of the tool point's x, y to each joint."""
        q = np.atleast_2d(q)
        a1 = q[:, 0] + self.zero
        a12 = a1 + q[:, 1]
        J = np.zeros((len(q), 2, 4))
        J[:, 0, 0] = -self.l1 * np.sin(a1) - self.l2 * np.sin(a12)
        J[:, 1, 0] = self.l1 * np.cos(a1) + self.l2 * np.cos(a12)
        J[:, 0, 1] = -self.l2 * np.sin(a12)
        J[:, 1, 1] = self.l2 * np.cos(a12)
        return J

    def heading_sensitivity(self) -> np.ndarray:
        """d heading / d joint: the rake turns with q1, q2 and q4."""
        return np.array([1.0, 1.0, 0.0, 1.0])

    # ---- statics / dynamics
    def gravity_torques(self, q: np.ndarray) -> np.ndarray:
        """Joint torques (N m) to hold the arm still. Vertical axes carry no gravity torque;
        the lift carries the tool's weight as a force (N) in the z slot."""
        q = np.atleast_2d(q)
        out = np.zeros((len(q), 4))
        out[:, 2] = self.mt * G
        return out

    def inertia_about_base(self, q: np.ndarray) -> np.ndarray:
        """Moment of inertia (kg m^2) about the base axis, links as uniform rods, tool as a point."""
        q = np.atleast_2d(q)
        l1, l2 = self.l1 / 1000, self.l2 / 1000
        r_tool = np.hypot(*(self.fk(q)[:, :2] - self.base).T) / 1000
        c2 = np.cos(q[:, 1])
        r2c2 = l1**2 + (l2 / 2) ** 2 + l1 * l2 * c2        # distance^2 of link 2's centre
        return self.m1 * l1**2 / 3 + self.m2 * (l2**2 / 12 + r2c2) + self.mt * r_tool**2


# ============================================================================ articulated
@dataclass
class Articulated:
    base: np.ndarray
    zero: float                 # tray-frame direction of q1 = 0 (rad)
    h0: float                   # shoulder axis above the sand
    lu: float
    lf: float
    wrist_drop: float
    lim1: tuple[float, float]
    lim2: tuple[float, float]
    lim3: tuple[float, float]
    mu: float
    mf: float
    mt: float
    n_joints: int = 5
    joint_names = ("q1 base", "q2 shoulder", "q3 elbow", "q4 wrist pitch", "q5 tool yaw")

    @classmethod
    def from_config(cls, arm: Arm) -> "Articulated":
        c = arm.articulated
        return cls(np.asarray(arm.base, float), math.radians(arm.zero_deg), c.shoulder_height, c.upper, c.fore, c.wrist_drop,
                   tuple(np.radians(c.j1_limits)), tuple(np.radians(c.j2_limits)),
                   tuple(np.radians(c.j3_limits)), c.upper_mass, c.fore_mass, c.tool_mass)

    def _planar(self, q):
        q = np.atleast_2d(q)
        a2, a23 = q[:, 1], q[:, 1] + q[:, 2]
        rho = self.lu * np.cos(a2) + self.lf * np.cos(a23)            # wrist, horizontal from the base axis
        hz = self.h0 + self.lu * np.sin(a2) + self.lf * np.sin(a23)   # wrist height above the sand
        return rho, hz

    def fk(self, q: np.ndarray) -> np.ndarray:
        q = np.atleast_2d(q)
        rho, hz = self._planar(q)
        a1 = q[:, 0] + self.zero
        x = self.base[0] + rho * np.cos(a1)
        y = self.base[1] + rho * np.sin(a1)
        # The tool's absolute pitch is q2 + q3 + q4; straight down is -pi/2, so "tilt" below is
        # its lean away from vertical, outwards positive. With the planned q4 the tilt is zero
        # and the tool point sits wrist_drop straight below the wrist.
        tilt = q[:, 1] + q[:, 2] + q[:, 3] + math.pi / 2
        x += self.wrist_drop * np.sin(tilt) * np.cos(a1)
        y += self.wrist_drop * np.sin(tilt) * np.sin(a1)
        z = hz - self.wrist_drop * np.cos(tilt)
        return np.column_stack([x, y, z, a1 + q[:, 4]])

    def elbows(self, q: np.ndarray) -> np.ndarray:
        q = np.atleast_2d(q)
        rho = self.lu * np.cos(q[:, 1])
        a1 = q[:, 0] + self.zero
        return np.column_stack([self.base[0] + rho * np.cos(a1), self.base[1] + rho * np.sin(a1),
                                self.h0 + self.lu * np.sin(q[:, 1])])

    def ik(self, poses: np.ndarray, q_prev: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        poses = np.atleast_2d(poses)
        d = poses[:, :2] - self.base
        rho = np.hypot(d[:, 0], d[:, 1])
        q1 = _wrap(np.arctan2(d[:, 1], d[:, 0]) - self.zero)
        h = poses[:, 2] + self.wrist_drop - self.h0
        c3 = (rho**2 + h**2 - self.lu**2 - self.lf**2) / (2 * self.lu * self.lf)
        ok = (np.abs(c3) <= 1.0) & (rho > 1e-6)
        q3 = -np.arccos(np.clip(c3, -1, 1))                           # elbow up
        q2 = np.arctan2(h, rho) - np.arctan2(self.lf * np.sin(q3), self.lu + self.lf * np.cos(q3))
        q4 = -math.pi / 2 - q2 - q3
        q5 = _continuous(poses[:, 3] - (q1 + self.zero), None if q_prev is None else q_prev[4])
        ok &= _within(q1, self.lim1) & _within(q2, self.lim2) & _within(q3, self.lim3)
        return np.column_stack([q1, q2, q3, q4, q5]), ok

    def in_limits(self, q: np.ndarray) -> np.ndarray:
        """Base, shoulder and elbow limits; the wrist pitch follows the elbow and the tool yaw
        turns without limit (slip ring)."""
        q = np.atleast_2d(q)
        return _within(q[:, 0], self.lim1) & _within(q[:, 1], self.lim2) & _within(q[:, 2], self.lim3)

    def jacobian_xy(self, q: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        q = np.atleast_2d(q).astype(float)
        J = np.zeros((len(q), 2, 5))
        for k in range(5):
            dq = np.zeros(5)
            dq[k] = eps
            J[:, :, k] = (self.fk(q + dq)[:, :2] - self.fk(q - dq)[:, :2]) / (2 * eps)
        return J

    def heading_sensitivity(self) -> np.ndarray:
        return np.array([1.0, 0.0, 0.0, 0.0, 1.0])

    def gravity_torques(self, q: np.ndarray) -> np.ndarray:
        """Holding torques (N m) at the shoulder, elbow and wrist pitch; links as uniform rods,
        the tool as a point mass hanging wrist_drop below the wrist (its centre of mass at the
        wrist axis horizontally, since the tool hangs vertically)."""
        q = np.atleast_2d(q)
        a2, a23 = q[:, 1], q[:, 1] + q[:, 2]
        lu, lf = self.lu / 1000, self.lf / 1000
        x_u = lu / 2 * np.cos(a2)
        x_e = lu * np.cos(a2)
        x_f = x_e + lf / 2 * np.cos(a23)
        x_w = x_e + lf * np.cos(a23)
        out = np.zeros((len(q), 5))
        out[:, 1] = G * (self.mu * x_u + self.mf * x_f + self.mt * x_w)
        out[:, 2] = G * (self.mf * (x_f - x_e) + self.mt * (x_w - x_e))
        return out


# ============================================================================ helpers
def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _within(a, lim):
    return (a >= lim[0] - 1e-9) & (a <= lim[1] + 1e-9)


def _continuous(a: np.ndarray, start: float | None) -> np.ndarray:
    """Unwrap a joint angle sequence, beginning at the turn nearest ``start``."""
    a = np.unwrap(np.asarray(a, float))
    if start is not None and len(a):
        a = a + 2 * math.pi * round((start - a[0]) / (2 * math.pi))
    return a


def make_arm(garden: Garden, kind: str | None = None):
    if garden.arm is None:
        raise ValueError("this garden has no arm")
    kind = kind or garden.arm.kind
    if kind == "scara":
        return Scara.from_config(garden.arm)
    if kind == "articulated":
        return Articulated.from_config(garden.arm)
    raise ValueError(f"unknown arm kind {kind!r}")


# ============================================================================ error budget
@dataclass(frozen=True)
class Drive:
    """Joint-output uncertainty of one drive type, in degrees.

    backlash: dead band the joint can sit anywhere in when the load reverses.
    resolution: smallest controllable step at the joint output (encoder or microstep).
    """
    name: str
    backlash: float
    resolution: float


def line_wobble(arm, q: np.ndarray, headings: np.ndarray, drive: Drive, tine_offsets,
                preloaded=None) -> tuple[np.ndarray, np.ndarray]:
    """Worst-case lateral (across-the-groove) position error of the tool point and of the outer
    tines, per pose, for joints each off by up to +-(backlash + resolution) / 2.

    Only the component across the direction of travel matters: an error along the groove just
    shifts where it is cut, an error across it bends the line. Joints listed in ``preloaded``
    carry a load that never reverses (gravity on a pitch joint), so their gears stay on one
    flank and only the resolution counts."""
    J = arm.jacobian_xy(q)                                   # (N, 2, n)
    n = np.column_stack([-np.sin(headings), np.cos(headings)])
    lateral = np.abs(np.einsum("ni,nij->nj", n, J))          # (N, n) mm per rad
    pre = np.zeros(J.shape[2], bool) if preloaded is None else np.asarray(preloaded, bool)
    e = np.radians(np.where(pre, drive.resolution, drive.backlash + drive.resolution) / 2)
    centre = (lateral * e).sum(axis=1)
    # A heading error swings the outer tine by offset * dheading, mostly along the path, but
    # the along-path component is harmless; its across component is offset * dheading * sin(0)=0
    # for a straight bar, so only the centre error remains across the groove. Report the
    # heading error separately as the along-path smear of the outer tines.
    dh = (np.abs(arm.heading_sensitivity()) * e).sum()
    smear = max(abs(o) for o in tine_offsets) * dh * np.ones(len(q))
    return centre, smear


def gravity_preloaded(arm, q: np.ndarray, margin: float = 0.05) -> np.ndarray:
    """Joints whose holding torque keeps one sign (and is at least ``margin`` N m) over all q."""
    tau = arm.gravity_torques(q)
    return ((tau > margin).all(axis=0) | (tau < -margin).all(axis=0))
