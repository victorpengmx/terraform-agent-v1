"""Benchmark file (DESIGN §16): loader, validator, coverage matrix and the demo sanity check.

    python -m eval.benchmark [--suite eval/benchmark.yaml] [--demos]

Each entry's `graph` is the answer key (an `agent.spec.FeatureGraph`), used only for grading; the agent never
sees it. Feature fields left out of the YAML are filled with their "unspecified" value (null, false, "")
before strict validation, so a mistyped key is still an error.

`--demos` grades the hand-scripted demos (`scripts/demos.py`) against the prompts they fit, with the eval's
matcher-mode checklist (`eval.match.match_checklist`). A good demo failing a major item means the answer key
is stricter than the prompt.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, get_args

import yaml
from pydantic import ValidationError

from agent.checklist import Item
from agent.spec import FeatureGraph, FeatureType, RelationKind, SpecFeature
from terrain.schema import Strict

SUITE = Path(__file__).parent / "benchmark.yaml"
ID_PATTERN = re.compile(r"^[ti]\d\d$")  # t = text, i = image
MIN_FEATURES, MAX_FEATURES = 3, 6
EXPECTED_COUNTS = {"text": 10, "image": 5, "regression": 5}
SPEC_ABLATION = {f"t{i:02d}" for i in range(1, 8)} | {"i01", "i02", "i03"}
DEMO_FITS = {"t01": "valley", "t02": "island", "t03": "mesa"}  # prompt id -> scripts/demos.py demo it fits

POSITION_DEFAULTS = {"x": None, "y": None, "image_x": None, "depth": None}
FEATURE_DEFAULTS = {"span_x": None, "span_y": None, "slope": None, "winding": None, "tallest": False, "part_of": None,
                    "notes": ""}
GRAPH_DEFAULTS = {"relations": [], "materials": "", "unsupported": []}


class BenchmarkEntry(Strict):
    id: str
    prompt: str | None = None
    references: list[str] = []  # image paths, relative to the suite file's folder
    source: str | None = None  # image entries: where the image comes from
    license: str | None = None
    tags: list[str]  # the feature types in the graph
    spec_ablation: bool
    regression: bool
    graph: FeatureGraph  # the answer key

    @property
    def kind(self) -> str:
        return "image" if self.references else "text"


class BenchmarkError(ValueError):
    """The file can't be loaded: bad YAML, or an entry that doesn't fit the format."""


def _filled(graph: Any) -> Any:
    if not isinstance(graph, dict):
        return graph
    features = [FEATURE_DEFAULTS | f | {"position": POSITION_DEFAULTS | (f.get("position") or {})} if isinstance(f, dict) else f
                for f in graph.get("features", [])]
    return GRAPH_DEFAULTS | graph | {"features": features}


def load(path: Path = SUITE) -> list[BenchmarkEntry]:
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise BenchmarkError(f"{path}: {e}") from None
    if not isinstance(raw, dict) or not isinstance(raw.get("prompts"), list):
        raise BenchmarkError(f"{path}: expected a mapping with a 'prompts' list")
    entries = []
    for k, item in enumerate(raw["prompts"]):
        name = item.get("id", f"entry {k}") if isinstance(item, dict) else f"entry {k}"
        try:
            entries.append(BenchmarkEntry.model_validate(item | {"graph": _filled(item.get("graph"))}))
        except (ValidationError, TypeError) as e:
            raise BenchmarkError(f"{name}: {e}") from None
    return entries


def missing_references(entries: list[BenchmarkEntry], root: Path = SUITE.parent) -> list[str]:
    """Reference files not on disk. Images without a redistribution license are gitignored, so a fresh clone
    lacks them until they are downloaded from the entry's source."""
    return [f"{e.id}: reference {r} not found; download it from {e.source}" for e in entries for r in e.references
            if not (root / r).is_file()]


def problems(entries: list[BenchmarkEntry], root: Path = SUITE.parent) -> list[str]:
    """Everything that makes the benchmark unusable or off-spec (empty = fine)."""
    out = missing_references(entries, root)
    ids = [e.id for e in entries]
    out += [f"duplicate id {i!r}" for i in sorted({i for i in ids if ids.count(i) > 1})]
    for e in entries:
        p = f"{e.id}: "
        if not ID_PATTERN.match(e.id):
            out.append(p + "id must be t01-t99 (text) or i01-i99 (image)")
        elif e.id.startswith("i") != (e.kind == "image"):
            out.append(p + ("an image entry needs references" if e.id.startswith("i") else "a text entry has references"))
        if e.kind == "text" and not (e.prompt or "").strip():
            out.append(p + "a text entry needs a prompt")
        if e.kind == "image" and not (e.source and e.license):
            out.append(p + "an image entry needs source and license")
        out += [p + x for x in e.graph.problems()]
        if not MIN_FEATURES <= len(e.graph.features) <= MAX_FEATURES:
            out.append(p + f"{len(e.graph.features)} features; the answer key keeps {MIN_FEATURES}-{MAX_FEATURES}")
        for f in e.graph.features:
            pos = f.position
            if e.kind == "text" and (pos.image_x or pos.depth):
                out.append(p + f"{f.id} has an image position (image_x/depth) in a text entry")
            if e.kind == "image" and (pos.x or pos.y):
                out.append(p + f"{f.id} has a text position (x/y) in an image entry")
        types = sorted({f.type for f in e.graph.features})
        if sorted(e.tags) != types:
            out.append(p + f"tags {sorted(e.tags)} != the graph's feature types {types}")
    kinds = [e.kind for e in entries]
    counts = {"text": kinds.count("text"), "image": kinds.count("image"), "regression": sum(e.regression for e in entries)}
    out += [f"{n} {k} entries; expected {EXPECTED_COUNTS[k]}" for k, n in counts.items() if n != EXPECTED_COUNTS[k]]
    ablation = {e.id for e in entries if e.spec_ablation}
    if ablation != SPEC_ABLATION:
        out.append(f"spec_ablation is set on {sorted(ablation)}; expected {sorted(SPEC_ABLATION)}")
    return out


# ---------------------------------------------------------------- coverage


def _row(f: SpecFeature) -> str:
    if f.type != "river":
        return f.type
    return {"almost straight": "river, straight", None: "river, any winding"}.get(f.winding, "river, winding+")


ROWS = [r for t in get_args(FeatureType)
        for r in (("river, straight", "river, winding+", "river, any winding") if t == "river" else (t,))]


def coverage(entries: list[BenchmarkEntry]) -> str:
    """Feature types and relations x prompt ids (cell = how many), with the number of prompts per row."""
    ids = [e.id for e in entries]
    lines = [f"{len(entries)} prompts: {sum(e.kind == 'text' for e in entries)} text, "
             f"{sum(e.kind == 'image' for e in entries)} image; spec_ablation {sum(e.spec_ablation for e in entries)}, "
             f"regression {sum(e.regression for e in entries)}"]

    def table(title: str, rows: list[str], cell) -> None:
        width = max(len(r) for r in rows) + 2
        lines.append("")
        lines.append(f"{title:<{width}}" + "".join(f"{i:>4}" for i in ids) + "  prompts")
        for r in rows:
            counts = [cell(e, r) for e in entries]
            lines.append(f"{r:<{width}}" + "".join(f"{(c or '.'):>4}" for c in counts) + f"  {sum(c > 0 for c in counts):>7}")

    table("feature", ROWS, lambda e, r: sum(_row(f) == r for f in e.graph.features))
    table("relation", [*get_args(RelationKind), "tallest", "part_of"],
          lambda e, r: (sum(f.tallest for f in e.graph.features) if r == "tallest" else
                        sum(f.part_of is not None for f in e.graph.features) if r == "part_of" else
                        sum(x.relation == r for x in e.graph.relations)))
    return "\n".join(lines)


# ---------------------------------------------------------------- demo sanity check


def demo_results(entries: list[BenchmarkEntry]) -> dict[str, tuple[str, list[Item], dict[str, list[str]]]]:
    """prompt id -> (demo name, matcher-mode checklist, matched ops per answer-key feature)."""
    from eval.match import match, match_checklist
    from scripts.demos import build

    by_id = {e.id: e for e in entries}
    out = {}
    for pid, demo in DEMO_FITS.items():
        if pid in by_id:
            stack = build(demo, seed=1)
            graph = by_id[pid].graph
            out[pid] = (demo, match_checklist(graph, stack), match(graph, stack).ops)
    return out


def failed_major(items: list[Item]) -> list[Item]:
    return [i for i in items if i.status == "fail" and i.severity == "major"]


def demo_report(results: dict[str, tuple[str, list[Item], dict[str, list[str]]]]) -> str:
    lines = []
    for pid, (demo, items, ops) in results.items():
        passed = sum(i.status == "pass" for i in items)
        lines.append(f"{pid} vs demo '{demo}': {passed}/{len(items)} items pass, {len(failed_major(items))} major failures")
        lines.append("  matched: " + "; ".join(f"{fid} <- {', '.join(o) or '(none)'}" for fid, o in ops.items()))
        lines += [f"  {i.status.upper()} [{i.severity}] {i.id}: {i.detail}" for i in items if i.status != "pass"]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", type=Path, default=SUITE)
    ap.add_argument("--demos", action="store_true", help="grade the fitting hand-scripted demos (a few seconds)")
    args = ap.parse_args()
    try:
        entries = load(args.suite)
    except BenchmarkError as e:
        sys.exit(f"cannot load the benchmark: {e}")
    print(coverage(entries))
    found = problems(entries, args.suite.parent)
    print("\nproblems:" if found else "\nno problems")
    for p in found:
        print(f"  - {p}")
    bad_demo = False
    if args.demos:
        results = demo_results(entries)
        print("\ndemo sanity check (matcher-mode checklist):")
        print(demo_report(results))
        bad_demo = any(failed_major(items) for _, items, _ in results.values())
    sys.exit(1 if found or bad_demo else 0)


if __name__ == "__main__":
    main()
