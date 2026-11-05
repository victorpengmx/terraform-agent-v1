# Evaluation report: opus_images

Generated 2026-11-05 from `eval/results/opus_images`: 5 finished runs, code 4add2fb.

Success = no failed major answer-key item (matcher mode) AND no failed blocking/major programmatic check AND judge pass. The checklist pass rate is reported as a continuous score.

## E1: Critic on vs off

No paired runs of `critic_on` and `critic_off` in these results.

## E2: SPEC on vs off

No paired runs of `critic_on` and `spec_off` in these results.

## E4: Cost and efficiency

| metric | `critic_on` |
|---|---|
| runs / successes | 5 / 0 |
| agent cost (builder, SPEC, critic) | $4.84 |
| judge cost | $0.027 |
| cost per successful build | n/a (0 successes) |
| tokens per successful build | n/a (0 successes) |
| latency p50 / p95 (s) | 385 / 449 |
| model calls per run | 21.4 |
| tool-error rate | 4% |
| critic fix validity | 98% |
| rollback rate (revisions) | 50% |
| comparisons | 14 |
| revision win rate | 36% |
| Terrain 1 win rate (decisive) | 60% |
| stop reasons | max_rounds 5 |

Tokens include cache reads and writes. Cost per successful build divides all of a condition's agent cost by its successes. The revision win rate should stay clearly below 100% and Terrain 1 near 50% (DESIGN §11.4). Latency is wall time per run with 3 run(s) at a time.

## E5: SPEC quality (agent graph vs answer key)

| prompts | runs | feature recall | precision | relation recall | category accuracy (matched features) |
|---|---|---|---|---|---|
| all | 5 | 100% | 57% | 58% | slope 100% exact / 100% ±1 (n=11); winding 100% exact / 100% ±1 (n=1); span_x 100% exact / 100% ±1 (n=2) |
| text | 0 | n/a | n/a | n/a | n/a |
| image | 5 | 100% | 57% | 58% | slope 100% exact / 100% ±1 (n=11); winding 100% exact / 100% ±1 (n=1); span_x 100% exact / 100% ±1 (n=2) |

Features match one-to-one on compatible type and overlapping position bands (eval/spec_quality.py). Category accuracy counts only categories the answer key states; one the agent left unspecified counts as wrong.

## E3: Judge calibration

Not scored yet (`python -m eval.calibrate score NAME`).

## Limitations

- Small N: 5 prompts (0 text, 5 image), seeds [1], 5 finished runs. Differences smaller than the CIs above can't be told apart from noise.
- Models (5 runs): builder `claude-opus-5-5`, critic `claude-opus-5-5`, judge `gemini-3.8-flash`.
- The judge is not yet calibrated against human ratings (E3 pending); its pass/fail is part of success.
- The answer key grades layout and categories only; looks rest on one judge model.
