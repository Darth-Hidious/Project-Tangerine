"""The command line on the proof-of-concept config: a planned program survives its own check,
and the check catches an unmodelled code, a joint beyond its limit and a head driven into a stone."""

import json
import re

import numpy as np
import pytest

from karesansui import cli, gcode
from karesansui.config import DEFAULT_CONFIG, poc_garden
from karesansui.planner import ArmMachine


def run(capsys, argv):
    capsys.readouterr()                                   # drop anything printed before this call
    code = cli.main([str(a) for a in argv])
    return code, json.loads(capsys.readouterr().out)


@pytest.fixture(scope="module")
def ripples(tmp_path_factory):
    path = tmp_path_factory.mktemp("cli") / "ripples.gcode"
    assert cli.main(["plan", "ripples", "-o", str(path)]) == 0
    return path.read_text().splitlines()


def write(tmp_path, lines):
    path = tmp_path / "edited.gcode"
    path.write_text("\n".join(lines) + "\n")
    return path


def axes_of(line):
    return np.array([float(v) for _, v in re.findall(r"([XYZA])(-?\d+(?:\.\d*)?)", line)])


def test_planned_program_passes_its_check_and_rakes_the_sand(ripples, tmp_path, capsys):
    png = tmp_path / "top.png"
    code, out = run(capsys, ["check", write(tmp_path, ripples), "--png", png])
    assert code == 0, out["errors"]
    assert out["joint_limit_violations"] == 0 and out["unsafe_samples"] == 0
    assert abs(out["sim"]["volume_error_rel"]) < 1e-9
    assert out["sim"]["relief_mm"]["std"] > 0.3                         # grooves were cut
    assert png.stat().st_size > 10_000


def test_plan_refuses_gcode_for_a_config_without_an_arm(tmp_path, capsys):
    target = tmp_path / "gantry.gcode"
    code, out = run(capsys, ["plan", "lines", "--config", DEFAULT_CONFIG, "-o", target])
    assert code == 2 and "no [arm]" in out["error"]
    assert not target.exists()


def test_check_rejects_an_unmodelled_code(ripples, tmp_path, capsys):
    lines = list(ripples)
    lines.insert(3, "G2 X10 Y10 I5 J0")                                  # an arc the model does not simulate
    code, out = run(capsys, ["check", write(tmp_path, lines)])
    assert code == 1 and "unsupported" in out["errors"][0]


def test_check_catches_a_joint_beyond_its_limit(ripples, tmp_path, capsys):
    lim = poc_garden().arm.scara.j1_limits[1]
    lines = list(ripples)
    k = next(i for i, ln in enumerate(lines) if ln.startswith("G0"))
    lines[k] = re.sub(r"X-?\d+(?:\.\d*)?", f"X{lim + 5:.4f}", lines[k])  # base yaw 5 deg past its stop
    code, out = run(capsys, ["check", write(tmp_path, lines)])
    assert code == 1 and out["joint_limit_violations"] >= 1


def test_check_catches_the_head_driven_into_a_stone(ripples, tmp_path, capsys):
    garden = poc_garden()
    machine = ArmMachine(garden, "scara")
    lines = list(ripples)
    start = next(i for i, ln in enumerate(lines) if ln.startswith("; rake pass"))
    settle = next(i for i in range(start, len(lines)) if lines[i].startswith("G4"))
    k = settle + 40                                                      # a feed move mid-pass
    assert lines[k].startswith("G1")
    q_prev = gcode.from_axes(axes_of(lines[k]), "scara")
    heading = machine.arm.fk(q_prev)[0, 3]
    centre = np.mean(garden.stones[0].outline, axis=0)                   # middle of the main stone
    q, ok = machine.ik(np.array([[*centre, heading]]), machine.z_for("rake"), q_prev=q_prev)
    assert ok[0]
    axes = gcode.to_axes(q[0], "scara")
    feed = lines[k].split("F")[1]
    lines[k] = "G1 " + " ".join(f"{a}{v:.4f}" for a, v in zip("XYZA", axes)) + f" F{feed}"
    code, out = run(capsys, ["check", write(tmp_path, lines)])
    assert code == 1 and out["unsafe_samples"] > 0
    assert out["joint_limit_violations"] == 0                            # it is the clearance check that fires
