"""Judge calibration (E3, DESIGN §16): your ratings vs the eval judge's.

    python -m eval.calibrate sample eval/results/e1_critic [more folders] --name NAME [--n 30] [--screenshots 8]
    python -m eval.calibrate score NAME

`sample` picks finished, judged runs stratified by judge verdict x text/image x condition (seeded), and writes
eval/calibration/NAME/:
- sheet.html: per item (c01, c02, ...) the request, the references and the judge's renders, in shuffled order.
  The condition and the judge's verdict are hidden. Screenshot items show the compile command and the expected
  screenshot files instead of renders.
- ratings.csv: fill in naturalism, minecraft_quality, materials, reference_match (images only) 1-5, pass y/n.
- key.json: item -> run (don't open it before rating).
`score` reads ratings.csv and reports Spearman correlation (your mean score vs the judge's), pass/fail
agreement, Cohen's kappa and the confusion matrix, for all items and separately for render- and
screenshot-rated ones. It writes score.md, which `eval.report --calibration NAME` includes as E3.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from eval import judge as judging
from eval.benchmark import SUITE, load
from eval.report import done, load_rows
from eval.rows import read_rows
from terrain.opstack import OpStack

ROOT = Path("eval/calibration")
DIMS = ("naturalism", "minecraft_quality", "materials", "reference_match")
FIELDS = ["item", "source", *DIMS, "pass", "notes"]
WORLD = r"%APPDATA%\.minecraft\saves\TerraformEval"
ORIGIN_STEP = 320  # blocks between compiled screenshot items (areas are 256 wide)
SEED = 0


def stratified(rows: list[dict[str, Any]], n: int, seed: int = SEED) -> list[dict[str, Any]]:
    """Round-robin over strata (judge verdict, kind, condition), random within each, until n rows."""
    rng = random.Random(seed)
    strata: dict[tuple, list] = defaultdict(list)
    for r in rows:
        strata[(bool((r.get("judge") or {}).get("passed")), r["kind"], r["condition"])].append(r)
    for v in strata.values():
        rng.shuffle(v)
    picked = []
    while len(picked) < n and any(strata.values()):
        for k in sorted(strata):
            if strata[k] and len(picked) < n:
                picked.append(strata[k].pop())
    rng.shuffle(picked)
    return picked


def sample(folders: list[Path], name: str, n: int = 30, screenshots: int = 8, suite: Path = SUITE, root: Path = ROOT,
           render: judging.Renderer = judging.render_views, seed: int = SEED) -> Path:
    entries = {e.id: e for e in load(suite)}
    rows = [r for r in done(load_rows(folders)) if (r.get("judge") or {}).get("passed") is not None]
    picked = stratified(rows, n, seed)
    out = root / name
    if out.exists():
        raise FileExistsError(f"{out} exists; pick another name (ratings would be overwritten)")
    (out / "images").mkdir(parents=True)
    (out / "screenshots").mkdir()
    shot_items = set(random.Random(seed + 1).sample(range(len(picked)), min(screenshots, len(picked))))
    items, key = [], {}
    for k, r in enumerate(picked):
        item = f"c{k + 1:02d}"
        e = entries[r["prompt_id"]]
        refs = []
        for j, ref in enumerate(e.references):
            dest = out / "images" / f"{item}_ref{j + 1}{Path(ref).suffix}"
            shutil.copyfile(suite.parent / ref, dest)
            refs.append(dest.name)
        renders = []
        stack_path = Path(r["stack"])
        views = judging.views_for(bool(e.references))
        saved = [stack_path.parent / f"judge_{v}.png" for v in views]
        if not all(p.exists() for p in saved):
            stack = OpStack.from_json(stack_path.read_text(encoding="utf-8"))
            for (v, png), p in zip(render(stack, views), saved):
                p.write_bytes(png)
        for v, p in zip(views, saved):
            dest = out / "images" / f"{item}_{v}.png"
            shutil.copyfile(p, dest)
            renders.append(dest.name)
        shot = k in shot_items
        origin = (len([i for i in items if i["screenshot"]]) * ORIGIN_STEP, 0) if shot else None
        stack_copy = out / "stacks" / f"{item}.json"  # the run's own path names its condition
        stack_copy.parent.mkdir(exist_ok=True)
        shutil.copyfile(stack_path, stack_copy)
        items.append({"item": item, "prompt": e.prompt, "refs": refs, "renders": renders, "screenshot": shot,
                      "origin": origin, "stack": stack_copy.as_posix(), "has_refs": bool(e.references)})
        key[item] = {"folder": next(f.as_posix() for f in folders if f.name == r["folder"]), "key": r["key"], "prompt_id": r["prompt_id"], "kind": r["kind"], "condition": r["condition"],
                     "judge": r["judge"]}
    with (out / "ratings.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for i in items:
            w.writerow({"item": i["item"], "source": "screenshot" if i["screenshot"] else "render",
                        "reference_match": "" if i["has_refs"] else "n/a"})
    (out / "key.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
    (out / "sheet.html").write_text(sheet(name, items), encoding="utf-8")
    return out


def sheet(name: str, items: list[dict[str, Any]]) -> str:
    esc = html.escape
    cards = []
    for i in items:
        parts = [f"<h2>{i['item']}</h2>", f"<p class=req><b>Request:</b> {esc(i['prompt'] or '(no text; see the reference)')}</p>"]
        if i["refs"]:
            parts.append("<div class=row>" + "".join(f"<figure><img src='images/{r}'><figcaption>reference</figcaption></figure>"
                                                    for r in i["refs"]) + "</div>")
        if i["screenshot"]:
            x, z = i["origin"]
            parts.append(
                "<p><b>Rate from in-game screenshots.</b> Compile, then take two screenshots: from the south edge "
                "looking north, and from the south-west corner.</p>"
                f"<pre>python -m scripts.compile_world --world \"{WORLD}\" --stack {esc(i['stack'])} --origin {x} {z}</pre>"
                "<div class=row>" + "".join(f"<figure><img src='screenshots/{i['item']}_{k}.png' alt='screenshots/{i['item']}_{k}.png'>"
                                            f"<figcaption>screenshots/{i['item']}_{k}.png</figcaption></figure>" for k in (1, 2)) + "</div>")
        else:
            parts.append("<div class=row>" + "".join(f"<figure><img src='images/{r}'><figcaption>{r.split('_', 1)[1][:-4]}</figcaption></figure>"
                                                    for r in i["renders"]) + "</div>")
        cards.append("<section>" + "".join(parts) + "</section>")
    return f"""<!doctype html>
<html lang=en><head><meta charset=utf-8><meta name=viewport content="width=device-width, initial-scale=1">
<title>Judge calibration {esc(name)}</title>
<style>
:root {{ --bg: #fafaf8; --fg: #1d1d1b; --muted: #66665f; --card: #fff; --line: #ddd; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg: #161615; --fg: #ececea; --muted: #a3a39c; --card: #20201e; --line: #3a3a37; }} }}
body {{ background: var(--bg); color: var(--fg); font: 15px/1.5 system-ui, sans-serif; margin: 0 auto; max-width: 1100px; padding: 16px; }}
section {{ background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 12px 16px; margin: 16px 0; }}
.row {{ display: flex; flex-wrap: wrap; gap: 8px; }} figure {{ margin: 0; flex: 1 1 300px; }}
img {{ width: 100%; height: auto; border-radius: 4px; }} figcaption, .muted {{ color: var(--muted); font-size: 13px; }}
pre {{ white-space: pre-wrap; word-break: break-all; font-size: 12px; }}
</style></head><body>
<h1>Judge calibration: {esc(name)}</h1>
<p>Rate each item in <code>ratings.csv</code>, on looks only (whether each landform exists and where is measured separately):
<b>naturalism</b>, <b>minecraft_quality</b> (block-scale cleanliness), <b>materials</b> (surface blocks suit the
landforms and vary naturally), <b>reference_match</b> (image items only), each 1-5 (3 = acceptable, 5 = showcase),
and <b>pass</b> y/n: would you accept it as a good-looking landscape for the request? Don't open key.json until you're done.</p>
{''.join(cards)}
</body></html>
"""


def _num(v: str) -> float | None:
    v = (v or "").strip()
    return float(v) if v and v.lower() != "n/a" else None


def _yes(v: str) -> bool | None:
    v = (v or "").strip().lower()
    return True if v in ("y", "yes", "pass", "1", "true") else False if v in ("n", "no", "fail", "0", "false") else None


def agreement(pairs: list[tuple[float, float, bool, bool]]) -> dict[str, Any]:
    """pairs: (human mean, judge mean, human pass, judge pass)."""
    from scipy.stats import spearmanr

    n = len(pairs)
    out: dict[str, Any] = {"n": n, "spearman": None, "spearman_p": None, "agreement": None, "kappa": None,
                           "confusion": {"both pass": 0, "human pass, judge fail": 0, "human fail, judge pass": 0, "both fail": 0}}
    if not n:
        return out
    h, j = [p[0] for p in pairs], [p[1] for p in pairs]
    if n >= 3 and len(set(h)) > 1 and len(set(j)) > 1:
        rho, p = spearmanr(h, j)
        out["spearman"], out["spearman_p"] = float(rho), float(p)
    for _, _, hp, jp in pairs:
        out["confusion"][{(True, True): "both pass", (True, False): "human pass, judge fail",
                          (False, True): "human fail, judge pass", (False, False): "both fail"}[(hp, jp)]] += 1
    po = sum(hp == jp for _, _, hp, jp in pairs) / n
    ph, pj = sum(p[2] for p in pairs) / n, sum(p[3] for p in pairs) / n
    pe = ph * pj + (1 - ph) * (1 - pj)
    out["agreement"] = po
    out["kappa"] = (po - pe) / (1 - pe) if pe < 1 else None
    return out


def score(name: str, root: Path = ROOT) -> tuple[dict[str, dict[str, Any]], Path]:
    out = root / name
    key = json.loads((out / "key.json").read_text(encoding="utf-8"))
    groups: dict[str, list] = defaultdict(list)
    for rating in csv.DictReader((out / "ratings.csv").open(encoding="utf-8")):
        dims = [_num(rating[d]) for d in DIMS]
        human_pass = _yes(rating["pass"])
        if human_pass is None or not any(d is not None for d in dims):
            continue  # not rated yet
        k = key[rating["item"]]
        row = read_rows(Path(k["folder"]) / "runs.jsonl").get(k["key"], {})
        j = row.get("judge") or k["judge"]  # the latest verdict, else the one at sampling time
        if j.get("passed") is None:
            continue
        pair = (mean(d for d in dims if d is not None), j["mean"], human_pass, bool(j["passed"]))
        groups["all"].append(pair)
        groups[rating["source"]].append(pair)
    results = {g: agreement(groups.get(g, [])) for g in ("all", "render", "screenshot")}
    md = ["## E3: Judge calibration", "",
          f"Your ratings vs the judge on `{name}` (mean of the rated dimensions vs the judge's mean score; pass vs pass).", ""]
    md += ["| items | n | Spearman ρ (p) | pass/fail agreement | Cohen's κ | both pass / you pass, judge fails / you fail, judge passes / both fail |",
           "|---|---|---|---|---|---|"]
    for g, r in results.items():
        rho = "n/a" if r["spearman"] is None else f"{r['spearman']:.2f} ({r['spearman_p']:.3f})"
        agr = "n/a" if r["agreement"] is None else f"{r['agreement']:.0%}"
        kap = "n/a" if r["kappa"] is None else f"{r['kappa']:.2f}"
        c = r["confusion"]
        md.append(f"| {g} | {r['n']} | {rho} | {agr} | {kap} | {c['both pass']} / {c['human pass, judge fail']} / "
                  f"{c['human fail, judge pass']} / {c['both fail']} |")
    path = out / "score.md"
    path.write_text("\n".join(md) + "\n", encoding="utf-8")
    return results, path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("folders", type=Path, nargs="+")
    s.add_argument("--name", required=True)
    s.add_argument("--n", type=int, default=30)
    s.add_argument("--screenshots", type=int, default=8)
    c = sub.add_parser("score")
    c.add_argument("name")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if args.cmd == "sample":
        out = sample(args.folders, args.name, args.n, args.screenshots)
        print(f"wrote {out / 'sheet.html'} and {out / 'ratings.csv'}")
    else:
        _, path = score(args.name)
        print(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
