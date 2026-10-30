"""The benchmark file and its validator (eval/benchmark.py, DESIGN §16)."""

import copy
import warnings

import pytest
import yaml

from eval.benchmark import (DEMO_FITS, SPEC_ABLATION, BenchmarkError, coverage, demo_results, failed_major, load,
                            missing_references, problems)

PEAK = {"id": "peak_1", "type": "peak", "position": {"x": "left"}}
RIVER = {"id": "river_1", "type": "river", "winding": "almost straight"}
LAKE = {"id": "lake_1", "type": "lake"}


def text_entry(id="t01", **kw):
    return {"id": id, "prompt": "a peak, a river and a lake", "tags": ["lake", "peak", "river"],
            "spec_ablation": id in SPEC_ABLATION, "regression": False,
            "graph": {"features": [PEAK, RIVER, LAKE],
                      "relations": [{"subject": "river_1", "relation": "reaches", "object": "lake_1"}]}} | kw


def image_entry(id="i01", **kw):
    features = [{"id": "peak_1", "type": "peak", "position": {"image_x": "center", "depth": "far"}}, RIVER, LAKE]
    return text_entry(id, prompt=None, references=["references/a.jpg"], source="https://example.org/a",
                      license="CC BY 4.0", graph={"features": features}) | kw


def full_suite():
    """A suite that meets every count: 10 text, 5 image, the right spec_ablation set, 5 regression."""
    entries = [text_entry(f"t{i:02d}") for i in range(1, 11)] + [image_entry(f"i{i:02d}") for i in range(1, 6)]
    for e in entries[:5]:
        e["regression"] = True
    return entries


def write(tmp_path, entries):
    path = tmp_path / "benchmark.yaml"
    path.write_text(yaml.safe_dump({"prompts": entries}), encoding="utf-8")
    (tmp_path / "references").mkdir(exist_ok=True)
    (tmp_path / "references" / "a.jpg").write_bytes(b"jpg")
    return path


def found(tmp_path, entries):
    return problems(load(write(tmp_path, entries)), tmp_path)


def test_the_shipped_benchmark_is_valid():
    """Missing reference images only warn: the unlicensed ones are gitignored, so a fresh clone lacks them."""
    entries = load()
    missing = missing_references(entries)
    assert [p for p in problems(entries) if p not in missing] == []
    for m in missing:
        warnings.warn(m)


def test_a_missing_reference_names_where_to_download_it(tmp_path):
    suite = full_suite()
    suite[10]["references"] = ["references/gone.jpg"]
    assert missing_references(load(write(tmp_path, suite)), tmp_path) == [
        "i01: reference references/gone.jpg not found; download it from https://example.org/a"]


def test_a_well_formed_suite_has_no_problems(tmp_path):
    assert found(tmp_path, full_suite()) == []


def test_omitted_feature_fields_are_filled_as_unspecified(tmp_path):
    e = load(write(tmp_path, full_suite()))[0]
    lake = e.graph.features[2]
    assert (lake.position.x, lake.position.image_x, lake.span_x, lake.slope, lake.winding) == (None,) * 5
    assert (lake.tallest, lake.part_of, lake.notes) == (False, None, "")
    assert e.graph.materials == "" and e.graph.unsupported == []
    assert e.kind == "text" and load(write(tmp_path, full_suite()))[-1].kind == "image"


def test_a_mistyped_key_is_an_error_not_ignored(tmp_path):
    suite = full_suite()
    suite[0]["graph"]["features"][0] = PEAK | {"slop": "steep"}
    with pytest.raises(BenchmarkError, match="(?s)t01.*slop"):
        load(write(tmp_path, suite))
    suite = full_suite()
    suite[1]["regresion"] = True
    with pytest.raises(BenchmarkError, match="(?s)t02.*regresion"):
        load(write(tmp_path, suite))


@pytest.mark.parametrize("change, expected", [
    (lambda s: s[1].update(id="t01"), "duplicate id 't01'"),
    (lambda s: s[10].update(references=["references/missing.jpg"]), "i01: reference references/missing.jpg not found"),
    (lambda s: s[10].update(license=None), "i01: an image entry needs source and license"),
    (lambda s: s[0].update(references=["references/a.jpg"]), "t01: a text entry has references"),
    (lambda s: s[0].update(prompt=" "), "t01: a text entry needs a prompt"),
    (lambda s: s[0]["graph"]["features"].extend(
        {"id": f"hill_{k}", "type": "hills"} for k in range(4)), "t01: 7 features; the answer key keeps 3-6"),
    (lambda s: s[0]["graph"]["features"].__setitem__(0, PEAK | {"position": {"depth": "near"}}),
     "t01: peak_1 has an image position"),
    (lambda s: s[10]["graph"]["features"].__setitem__(0, PEAK), "i01: peak_1 has a text position"),
    (lambda s: s[0].update(tags=["peak", "river"]), "t01: tags ['peak', 'river'] != the graph's feature types"),
    (lambda s: s[0]["graph"]["relations"].append({"subject": "river_1", "relation": "flows_from", "object": "range_9"}),
     "t01: relation object 'range_9' is not a feature"),
    (lambda s: s[7].update(spec_ablation=True), "spec_ablation is set on"),
    (lambda s: s[6].update(regression=True), "6 regression entries; expected 5"),
    (lambda s: s.pop(), "4 image entries; expected 5"),
])
def test_bad_entries_are_flagged(tmp_path, change, expected):
    suite = copy.deepcopy(full_suite())
    change(suite)
    assert any(p.startswith(expected) for p in found(tmp_path, suite)), found(tmp_path, suite)


def test_coverage_counts_features_relations_and_river_winding(tmp_path):
    suite = full_suite()[:2]
    suite[1]["graph"]["features"][1] = RIVER | {"winding": "twisty"}
    text = coverage(load(write(tmp_path, suite)))
    rows = {line.split("  ")[0].strip(): line.split() for line in text.splitlines()[1:] if line.strip()}
    assert rows["peak"][-3:] == ["1", "1", "2"]
    assert rows["river, straight"][-3:] == ["1", ".", "1"]
    assert rows["river, winding+"][-3:] == [".", "1", "1"]
    assert rows["reaches"][-3:] == ["1", "1", "2"]
    assert rows["basin"][-1] == "0"


def test_each_fitting_demo_passes_its_prompts_answer_key():
    """A good hand-scripted build failing a major item would mean the answer key is stricter than the prompt."""
    results = demo_results(load())
    assert set(results) == set(DEMO_FITS)
    for pid, (demo, items, ops) in results.items():
        assert failed_major(items) == [], (pid, demo, failed_major(items))
        assert all(ops.values()), (pid, ops)
