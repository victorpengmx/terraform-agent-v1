import numpy as np
import pytest

from terrain.features import (SLOPE_CLASSES, WINDING_CLASSES, category_distance, classify, feature_report,
                              measure_features, peak_prominences, slack)
from terrain.opstack import BaseSpec, OpStack

BASE = 64.0


def stack(n=65, seed=1) -> OpStack:
    return OpStack(BaseSpec(n, n, base_y=BASE, seed=seed))


def test_classes_have_half_open_bounds():
    assert [classify(v, SLOPE_CLASSES) for v in (0, 14.9, 15, 35, 60, 89)] == \
        ["gentle", "gentle", "moderate", "steep", "cliff", "cliff"]
    assert [classify(v, WINDING_CLASSES) for v in (1.0, 1.05, 1.3, 2.0)] == \
        ["almost straight", "winding", "twisty", "meandering"]


def test_category_distance_and_slack():
    assert category_distance("gentle", "gentle", SLOPE_CLASSES) == 0
    assert slack(category_distance("gentle", "moderate", SLOPE_CLASSES)) == "minor"
    assert slack(category_distance("gentle", "cliff", SLOPE_CLASSES)) == "major"
    with pytest.raises(ValueError, match="unknown category"):
        category_distance("gentle", "vertical", SLOPE_CLASSES)


def test_prominence_of_one_and_two_peaks():
    h = np.full((5, 9), 10.0)
    h[2, 2], h[2, 6] = 30.0, 25.0  # two summits
    h[2, 3:6] = 18.0  # joined by a ridge whose col is 18
    prom = peak_prominences(h)
    assert prom[2 * 9 + 2] == pytest.approx(20.0)  # highest: above the area's lowest point
    assert prom[2 * 9 + 6] == pytest.approx(25.0 - 18.0)  # lower: above the col


def test_flat_top_is_one_peak():
    h = np.full((5, 5), 0.0)
    h[1:4, 1:4] = 10.0
    prom = peak_prominences(h)
    assert sorted(prom.values()) == [10.0]


def test_peak_attributes():
    s = stack()
    s.add("raise_peak", {"center": [0.5, 0.5], "radius": 0.3, "height": 40, "roughness": 0.0}, id="mountain")
    a = measure_features(s)["mountain"]
    assert a.type == "peak" and a.top_y == pytest.approx(BASE + 40)
    assert a.top_uv == pytest.approx((0.5, 0.5)) and a.prominence == pytest.approx(40, abs=0.5)
    assert a.edges == ()


def test_slope_class_tracks_steepness():
    gentle, steep = stack(), stack()
    gentle.add("raise_peak", {"center": [0.5, 0.5], "radius": 0.45, "height": 4, "character": "gentle"}, id="hill")
    steep.add("raise_peak", {"center": [0.5, 0.5], "radius": 0.2, "height": 60, "character": "jagged"}, id="spire")
    assert measure_features(gentle)["hill"].slope_class == "gentle"
    assert SLOPE_CLASSES[[n for n, _ in SLOPE_CLASSES].index(measure_features(steep)["spire"].slope_class)][1] > 35


def test_river_sinuosity_and_edges():
    river = {"path": [[0.5, 0.0], [0.5, 1.0]], "width": 0.06, "depth": 3}
    straight, winding = stack(), stack()
    straight.add("carve_river", river, id="river")
    winding.add("carve_river", river | {"meander": 0.8}, id="river")
    a, b = measure_features(straight)["river"], measure_features(winding)["river"]
    assert a.sinuosity == pytest.approx(1.0, abs=0.01) and a.winding == "almost straight"
    assert b.sinuosity > 1.05 and b.winding != "almost straight"
    assert set(a.edges) == {"n", "s"}


def test_modifiers_are_not_features_and_report_reads_well():
    s = stack()
    s.add("raise_peak", {"center": [0.3, 0.3], "radius": 0.2, "height": 30}, id="mountain")
    s.add("smooth", {"strength": 0.5})
    s.add("carve_river", {"path": [[0.8, 0.0], [0.8, 1.0]], "width": 0.05, "depth": 3}, id="river")
    attrs = measure_features(s)
    assert list(attrs) == ["mountain", "river"]
    report = feature_report(attrs)
    assert "mountain (peak" in report and "prominence" in report and "sinuosity" in report
    assert "reaches north, south edge" in report
