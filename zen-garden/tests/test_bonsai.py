"""The bonsai the viewer and the renders draw: grown inside the model's tree, and shaped like one."""

from dataclasses import replace

import numpy as np
import pytest
import shapely
from shapely.geometry import Polygon

from karesansui.bonsai import grow, tuft_mesh
from karesansui.config import poc_garden
from karesansui.landscape import terrain


@pytest.fixture(scope="module")
def grown():
    g = poc_garden()
    t = terrain(g, dx=2.0, overlays=False)
    ny, nx = t.ground.shape

    def ground_at(x, y):
        return float(t.ground[min(max(int(y / 2), 0), ny - 1), min(max(int(x / 2), 0), nx - 1)])
    return g, ground_at, grow(g, ground_at)


def canopy_of(g):
    return next(f for f in g.features if f.kind == "tree")


def test_every_needle_stays_inside_the_canopy_the_arm_keeps_clear_of(grown):
    g, _, tree = grown
    tips = tree.needle_tips()
    assert len(tips) > 1000
    assert shapely.contains_xy(Polygon(canopy_of(g).outline), tips[:, 0], tips[:, 1]).all()
    assert tips[:, 2].max() <= tree.base[2] + canopy_of(g).height


def test_it_is_shaped_like_a_bonsai(grown):
    g, _, tree = grown
    trunk, r = tree.branches[0]
    assert r[0] > 8 * r[-1]                                   # a strong taper to the apex
    assert np.all(np.diff(trunk[:, 2]) > 0)                  # the trunk rises all the way
    height = canopy_of(g).height
    lowest = tree.tuft_array[:, 2].min() - tree.base[2]
    assert 0.25 * height < lowest < 0.4 * height             # a bare lower trunk, the first pad at about a third
    spread = np.ptp(tree.needle_tips()[:, :2], axis=0)
    assert spread.min() > 0.5 * height                        # pads spread wide, not a column


def test_a_smaller_canopy_grows_a_smaller_tree(grown):
    # The canopy outline is what the generator fits to: shrink it, and every needle still fits
    g, ground_at, tree = grown
    outline = np.asarray(canopy_of(g).outline, float)
    c = outline.mean(0)
    small = [tuple(c + (p - c) * 0.6) for p in outline]
    g2 = replace(g, features=tuple(replace(f, outline=tuple(small)) if f.kind == "tree" else f for f in g.features))
    tips = grow(g2, ground_at).needle_tips()
    assert shapely.contains_xy(Polygon(small), tips[:, 0], tips[:, 1]).all()
    assert np.ptp(tips[:, 0]) < 0.8 * np.ptp(tree.needle_tips()[:, 0])


def test_the_needle_shoot_is_one_unit_long():
    v, f = tuft_mesh()
    assert f.max() < len(v)
    assert 0.8 < v[:, 2].max() <= 1.35 and np.abs(v[:, :2]).max() < 0.9
