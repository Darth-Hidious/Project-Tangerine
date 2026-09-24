"""Renderer physics against closed-form results: colour temperature, point-light illuminance, shadows."""

import math

import numpy as np
import pytest

from karesansui import render


def test_daylight_white_is_neutral_and_warm_light_is_warm():
    rgb = render.cct_to_rgb(6504.0, white=6504.0)
    assert np.allclose(rgb, 1.0, atol=0.02)                  # D65 on a D65-balanced camera: grey
    warm = render.cct_to_rgb(2200.0, white=6504.0)
    assert warm[0] > warm[1] > warm[2]
    lum = 0.2126 * warm[0] + 0.7152 * warm[1] + 0.0722 * warm[2]
    assert lum == pytest.approx(1.0)


def test_kang_locus_matches_published_chromaticity():
    # CIE illuminant A (2856 K) sits at x=0.4476, y=0.4074 on the Planckian locus.
    x, y = render.cct_to_xy(2856.0)
    assert x == pytest.approx(0.4476, abs=2e-3)
    assert y == pytest.approx(0.4074, abs=2e-3)


def _flat_scene(nx=200, ny=200, dx=1.0, light=(100.0, 100.0, 80.0), cd=10.0):
    h = np.zeros((ny, nx))
    albedo = np.full((ny, nx, 3), 0.5)
    lights = [(light[0], light[1], light[2], cd, np.ones(3), 0.0)]
    return render.Scene(h, albedo, np.zeros_like(h, bool), np.zeros_like(h), 0.0, 0.0, dx, lights)


def test_point_light_on_a_flat_floor_obeys_the_inverse_square_cosine_law():
    sc = _flat_scene()
    bk = render.bake(sc, ambient_lux=0.0)
    ny, nx = sc.h.shape
    X, Y = np.meshgrid((np.arange(nx) + 0.5), (np.arange(ny) + 0.5))
    lx, ly, lz, cd = 100.0, 100.0, 80.0, 10.0
    d2 = (X - lx) ** 2 + (Y - ly) ** 2 + lz**2
    expected = cd * lz / d2**1.5 * 1e6                       # cd * cos(theta) / d^2, mm -> lux
    assert np.allclose(bk.lux, expected, rtol=1e-9)


def test_a_wall_casts_a_shadow_and_a_lower_wall_does_not():
    sc = _flat_scene(light=(20.0, 100.0, 30.0))
    sc.h[:, 60:62] = 40.0                                    # taller than the light: shadow behind it
    lux_tall = render.bake(sc, ambient_lux=0.0).lux
    assert lux_tall[100, 150] == 0.0 and lux_tall[100, 40] > 0.0
    sc.h[:, 60:62] = 5.0                                     # a low kerb: the far floor stays lit
    lux_low = render.bake(sc, ambient_lux=0.0).lux
    assert lux_low[100, 150] > 0.0
