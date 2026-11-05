"""Recreate the M6 evaluation (DESIGN §16): each experiment as the exact commands that produce it.

    python -m eval.reproduce --list
    python -m eval.reproduce pilot_haiku pilot [--yes] [--print]
    python -m eval.reproduce all --yes

Steps run in order with this Python, stopping at the first failure. Paid steps (run_eval, regression) get
--yes only when you pass it; without it each prints its cost estimate and stops, so the first paid step ends
the sequence. Runs resume: finished runs are skipped, so re-running an experiment only pays for what's missing.
--print shows the commands without running anything.

Phase C order: pilot_haiku (pipeline check, every role on Haiku), pilot (the eval models), e1, e2, report;
then calibration_sample, your ratings, calibration_score. opus, opus_images and regression are optional.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

REPORT = "m6"  # report and calibration name for the full experiments


def run_eval(config: str) -> list[str]:
    return ["eval.run_eval", "--config", f"eval/configs/{config}.yaml", "--paid"]


def rescore(results: str) -> list[str]:
    return ["eval.rescore", f"eval/results/{results}"]


def report(name: str, *results: str, calibration: str | None = None) -> list[str]:
    return ["eval.report", *(f"eval/results/{r}" for r in results), "--name", name,
            *(["--calibration", calibration] if calibration else [])]


# experiment -> steps; each step is a module and its arguments ("--paid" marks where --yes goes).
EXPERIMENTS: dict[str, list[list[str]]] = {
    "validate": [["eval.benchmark", "--demos"]],
    "pilot_haiku": [run_eval("pilot_haiku"), rescore("pilot_haiku"), report("pilot_haiku", "pilot_haiku")],
    "pilot": [run_eval("pilot"), rescore("pilot"), report("pilot", "pilot")],
    "e1": [run_eval("e1_critic"), rescore("e1_critic")],
    "e2": [run_eval("e2_spec"), rescore("e2_spec")],
    "report": [report(REPORT, "e1_critic", "e2_spec")],
    "opus": [run_eval("opus"), rescore("opus"), report("opus", "opus")],
    "opus_images": [run_eval("opus_images"), rescore("opus_images"), report("opus_images", "opus_images")],
    "regression": [["eval.regression", "--paid"]],
    "calibration_sample": [["eval.calibrate", "sample", "eval/results/e1_critic", "eval/results/e2_spec", "--name", REPORT]],
    "calibration_score": [["eval.calibrate", "score", REPORT], report(REPORT, "e1_critic", "e2_spec", calibration=REPORT)],
}
ALL = ["validate", "pilot_haiku", "pilot", "e1", "e2", "report"]


def commands(names: list[str], yes: bool) -> list[tuple[list[str], bool]]:
    """(argv after `python -m`, paid) for these experiments' steps, in order; `all` expands to ALL. A paid
    step gets --yes only when `yes`."""
    out = []
    for name in names:
        for exp in ALL if name == "all" else [name]:
            if exp not in EXPERIMENTS:
                raise SystemExit(f"unknown experiment {exp!r}; choose from: all, {', '.join(EXPERIMENTS)}")
            for step in EXPERIMENTS[exp]:
                paid = "--paid" in step
                argv = [a for a in step if a != "--paid"] + (["--yes"] if paid and yes else [])
                out.append((argv, paid))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiments", nargs="*")
    ap.add_argument("--yes", action="store_true", help="let the paid steps spend (each still prints its estimate first)")
    ap.add_argument("--print", action="store_true", help="show the commands only")
    ap.add_argument("--list", action="store_true", help="list the experiments and their commands")
    args = ap.parse_args()
    if args.list or not args.experiments:
        for name in EXPERIMENTS:
            print(f"{name}:")
            for step, paid in commands([name], yes=False):
                print(f"  python -m {' '.join(step)}" + ("  (paid: spends only with --yes)" if paid else ""))
        print(f"all = {' + '.join(ALL)}")
        return
    for step, _ in commands(args.experiments, args.yes):
        print(f"$ python -m {' '.join(step)}", flush=True)
        if args.print:
            continue
        code = subprocess.run([sys.executable, "-m", *step]).returncode
        if code:
            sys.exit(f"stopped: `python -m {' '.join(step)}` exited with {code}"
                     + ("" if args.yes else " (paid steps need --yes)"))


if __name__ == "__main__":
    main()
