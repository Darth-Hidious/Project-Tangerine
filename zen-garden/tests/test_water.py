"""Water-loop formulas against reference values and limiting cases."""

import math

import pytest

from karesansui import water
from karesansui.config import poc_garden


def test_saturation_pressure_matches_steam_tables():
    assert water.p_sat_kpa(20.0) == pytest.approx(2.339, rel=5e-3)     # steam tables: 2.339 kPa
    assert water.p_sat_kpa(30.0) == pytest.approx(4.247, rel=5e-3)     # 4.247 kPa


def test_no_evaporation_into_saturated_air_and_linear_in_area():
    assert water.evaporation_l_per_day(1.0, t_water=20, t_air=20, rh=1.0) == 0.0
    one = water.evaporation_l_per_day(0.02)
    assert water.evaporation_l_per_day(0.04) == pytest.approx(2 * one)


def test_evaporation_hand_calculation():
    # A = 1 m^2, water 20 C, air 22 C / 45 %, V = 0.1 m/s
    pw, pa = 0.61094 * math.exp(17.625 * 20 / 263.04), 0.45 * 0.61094 * math.exp(17.625 * 22 / 265.04)
    kg_s = (pw - pa) * (0.089 + 0.00782) / (2501 - 2.361 * 20)
    assert water.evaporation_l_per_day(1.0) == pytest.approx(kg_s * 86400 / 998 * 1000)


def test_friction_is_laminar_below_transition_and_blasius_above():
    assert water.friction_factor(1000) == pytest.approx(0.064)
    assert water.friction_factor(10000) == pytest.approx(0.316 / 10)


def test_operating_point_lies_on_both_curves():
    q, h = water.operating_point(3.0, 4.0, 0.08, 0.006, 0.7)
    assert 0 < q < 4.0
    assert h == pytest.approx(3.0 * (1 - q / 4.0), rel=1e-9)
    assert h == pytest.approx(water.system_head_m(q, 0.08, 0.006, 0.7), rel=1e-6)


def test_film_depth_satisfies_nusselt_relation():
    h_mm, v, fr = water.film_depth_mm(1.0, 0.04, 0.1)
    h = h_mm / 1000
    q = 1.0 / 60000 / 0.04
    assert q == pytest.approx(9.81 * 0.1 * h**3 / (3 * 1e-6))
    assert v == pytest.approx(q / h)


def test_poc_water_report_is_plausible():
    r = water.report(poc_garden())
    assert 0.015 < r.open_water_m2 < 0.05
    lo, hi = r.evaporation_l_day
    assert 0.03 < lo < hi < 0.5
    assert r.hydraulic_power_w < 0.5
