"""The experiment scripts (eval/reproduce.py): every step names a real config or results folder, paid steps
spend only with --yes, and a step that stops (no --yes) ends the sequence."""

import subprocess
import sys
from pathlib import Path

import pytest

from eval import reproduce


def test_every_step_points_at_a_real_config_and_module():
    for name in reproduce.EXPERIMENTS:
        for argv, paid in reproduce.commands([name], yes=False):
            assert Path(argv[0].replace(".", "/") + ".py").exists(), argv
            for a in argv:
                if a.startswith("eval/configs/"):
                    assert Path(a).exists(), a
            assert paid == (argv[0] in ("eval.run_eval", "eval.regression")), argv


def test_paid_steps_get_yes_only_when_asked():
    without = reproduce.commands(["pilot"], yes=False)
    with_yes = reproduce.commands(["pilot"], yes=True)
    assert without[0][0] == ["eval.run_eval", "--config", "eval/configs/pilot.yaml"]
    assert with_yes[0][0] == ["eval.run_eval", "--config", "eval/configs/pilot.yaml", "--yes"]
    assert without[1:] == with_yes[1:]  # rescore and report are free: never --yes
    assert "--yes" not in reproduce.commands(["calibration_score"], yes=True)[0][0]


def test_all_runs_the_phase_c_order():
    names = [argv[2] if argv[0] == "eval.run_eval" else argv[0] for argv, _ in reproduce.commands(["all"], yes=False)]
    runs = [n for n in names if n.startswith("eval/configs/")]
    assert runs == ["eval/configs/pilot_haiku.yaml", "eval/configs/pilot.yaml", "eval/configs/e1_critic.yaml",
                    "eval/configs/e2_spec.yaml"]
    assert names[0] == "eval.benchmark" and names[-1] == "eval.report"


def test_an_unknown_experiment_is_an_error():
    with pytest.raises(SystemExit, match="unknown experiment 'e3'"):
        reproduce.commands(["e3"], yes=False)


def test_print_runs_nothing(monkeypatch, capsys):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("ran a command under --print"))
    monkeypatch.setattr(sys, "argv", ["reproduce", "e1", "e2", "--print", "--yes"])
    reproduce.main()
    out = capsys.readouterr().out
    assert "$ python -m eval.run_eval --config eval/configs/e1_critic.yaml --yes" in out
    assert "$ python -m eval.rescore eval/results/e2_spec" in out


def test_a_failing_step_stops_the_sequence(monkeypatch):
    ran = []

    def fake_run(argv, **kw):
        ran.append(argv[2:])
        return subprocess.CompletedProcess(argv, 1)  # run_eval without --yes exits 1 after its estimate

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["reproduce", "pilot"])
    with pytest.raises(SystemExit, match="paid steps need --yes"):
        reproduce.main()
    assert ran == [["eval.run_eval", "--config", "eval/configs/pilot.yaml"]]  # rescore and report never ran
