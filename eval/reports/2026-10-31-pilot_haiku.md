# Evaluation report: pilot_haiku

Generated 2026-10-31 from `eval/results/pilot_haiku`: 4 finished runs, code 0073377.

Success = no failed major answer-key item (matcher mode) AND no failed blocking/major programmatic check AND judge pass. The checklist pass rate is reported as a continuous score.

## E1: Critic on vs off

| prompts | condition | runs | success | judge unknown | mean checklist pass rate |
|---|---|---|---|---|---|
| all | `critic_on` | 2 | 50% | 0 | 75% |
| all | `critic_off` | 2 | 50% | 0 | 90% |
| text | `critic_on` | 1 | 100% | 0 | 100% |
| text | `critic_off` | 1 | 100% | 0 | 100% |
| image | `critic_on` | 1 | 0% | 0 | 50% |
| image | `critic_off` | 1 | 0% | 0 | 80% |

Paired by (prompt, seed): `critic_on` minus `critic_off`. Wins/losses/ties count pairs where `critic_on` is better/worse/equal; p is the exact two-sided sign test (ties dropped); the CI resamples prompts (10 000 bootstrap samples). No verdict below 5 prompts.

| prompts | metric | W / L / T | unknown | sign-test p | difference [95% CI] | verdict |
|---|---|---|---|---|---|---|
| all | success | 0 / 0 / 2 | 0 | 1.000 | +0 pts [+0 pts, +0 pts] | too few prompts to tell (2) |
| all | pass rate | 0 / 1 / 1 | 0 | 1.000 | -15 pts [-30 pts, +0 pts] | too few prompts to tell (2) |
| text | success | 0 / 0 / 1 | 0 | 1.000 | +0 pts [+0 pts, +0 pts] | too few prompts to tell (1) |
| text | pass rate | 0 / 0 / 1 | 0 | 1.000 | +0 pts [+0 pts, +0 pts] | too few prompts to tell (1) |
| image | success | 0 / 0 / 1 | 0 | 1.000 | +0 pts [+0 pts, +0 pts] | too few prompts to tell (1) |
| image | pass rate | 0 / 1 / 0 | 0 | 1.000 | -30 pts [-30 pts, -30 pts] | too few prompts to tell (1) |

Per prompt (success per seed: ✓ pass, ✗ fail, ? judge unknown; pass rate is the mean over seeds):

| prompt | kind | `critic_on` success | `critic_on` pass rate | `critic_off` success | `critic_off` pass rate |
|---|---|---|---|---|---|
| i01 | image | ✗ | 50% | ✗ | 80% |
| t01 | text | ✓ | 100% | ✓ | 100% |

## E2: SPEC on vs off

No paired runs of `critic_on` and `spec_off` in these results.

## E4: Cost and efficiency

| metric | `critic_off` | `critic_on` |
|---|---|---|
| runs / successes | 2 / 1 | 2 / 1 |
| agent cost (builder, SPEC, critic) | $0.011 | $0.047 |
| judge cost | $0.001 | $0.001 |
| cost per successful build | $0.011 | $0.047 |
| tokens per successful build | 379,512 | 1,008,472 |
| latency p50 / p95 (s) | 36 / 42 | 140 / 186 |
| model calls per run | 7.5 | 21.5 |
| tool-error rate | 0% | 0% |
| critic fix validity | n/a | 100% |
| rollback rate (revisions) | n/a | 25% |
| comparisons | 0 | 6 |
| revision win rate | n/a | 33% |
| Terrain 1 win rate (decisive) | n/a | 67% |
| stop reasons | done 2 | max_rounds 2 |

Tokens include cache reads and writes. Cost per successful build divides all of a condition's agent cost by its successes. The revision win rate should stay clearly below 100% and Terrain 1 near 50% (DESIGN §11.4).

## E5: SPEC quality (agent graph vs answer key)

| prompts | runs | feature recall | precision | relation recall | category accuracy (matched features) |
|---|---|---|---|---|---|
| all | 4 | 94% | 89% | 67% | slope 0% exact / 0% ±1 (n=2); winding 100% exact / 100% ±1 (n=2) |
| text | 2 | 100% | 100% | 100% | winding 100% exact / 100% ±1 (n=2) |
| image | 2 | 88% | 78% | 0% | slope 0% exact / 0% ±1 (n=2) |

Features match one-to-one on compatible type and overlapping position bands (eval/spec_quality.py). Category accuracy counts only categories the answer key states; one the agent left unspecified counts as wrong.

## E3: Judge calibration

Not scored yet (`python -m eval.calibrate score NAME`).

## Limitations

- Small N: 2 prompts (1 text, 1 image), seeds [1], 4 finished runs. Differences smaller than the CIs above can't be told apart from noise.
- Models (2 runs): builder `claude-haiku-5-5`, critic `claude-haiku-5-5`, judge `claude-haiku-5-5`.
- Models (2 runs): builder `claude-haiku-5-5`, critic `off`, judge `claude-haiku-5-5`.
- The judge shares the builder's model family in 4 runs. Acceptable for E1 and E2: both arms of each comparison use the same builder, so a family preference affects them alike.
- The judge shares the critic's model family in 2 runs: it may share the taste the critic optimized for, which favours critic_on in E1.
- The judge is not yet calibrated against human ratings (E3 pending); its pass/fail is part of success.
- The answer key grades layout and categories only; looks rest on one judge model.
