# Evidence log

One row per tool, prompt, critic or harness change, including reverted experiments. Cite traces, runs or
renders. Newest last.

| Date | Change | Evidence | Outcome |
|---|---|---|---|
| 2026-10-22 | Guardrail (§5.8): one agent call may move at most `MAX_MEAN_DH` = 40 blocks of height per column on average; `update_op` included, `remove_op` exempt | Per-op sum \|Δh\| / columns over the three demos: largest 26.0 (island seafloor, whole area −26), next 11.25 (island), 6.05 (valley peak); all others < 6 | Kept. 1.5× margin over the largest legitimate edit; tests in `tests/agent/test_guardrail.py` |
| 2026-10-22 | Op paths spline-smoothed (§5.2): `carve_valley` and `carve_river` now use the region's Catmull-Rom curve (`brushes.smooth_path`) | `renders/evidence/smoothing_{valley,mesa}_before_after.png`: the mesa canyon's corners at u≈0.35 and 0.55 are rounded; valley lake and river still connect. Test: 2 blocks inside a V-shaped path the raw polyline cut 16.2 blocks deep, the curve 6.6 | Kept |
| 2026-10-22 | River channel gets a flat floor half a sample step (0.25 block) wide | Smoothing moved samples off integer columns, and `test_river_bed_descends_monotonically` failed: the bed rose by up to 0.015 between samples (parabolic cross-section around the nearest 0.5-block sample). Diagonal and meandered rivers already had these bumps; the old straight test path hid them | Kept; bed monotonic again |
| 2026-10-22 | `update_op` accepts JSON-encoded `params`; other non-objects are an agent-facing error. The harness traces the calls of a step that crashes before re-raising | First Haiku run `traces/20261022-133534-598ae6.jsonl` crashed in round 2 with `ValueError` in the `update_op` handler (params sent as a string), and the crashing call was missing from the trace | Kept; test in `test_control_tools_list_update_remove` |
| 2026-10-22 | M3 milestone / M4 baseline: Haiku 5.5 builder and critic, 3 rounds, L1 renders, score-gated acceptance | `traces/20261022-133743-e5a431.jsonl`, $0.024, 48 tool calls, 0 errors. Scores 3.00 → 3.00 (**rolled back on a tie**: "score 3.00 <= best 3.00", although the revision had addressed the critique) → 3.50 (kept). Critic repeated "smooth round dome" for both mountains every round. Compiled into `TerraformTest` at (900, 0), snapshot 0021 | Baseline for M4. The tie rollback is the failure mode §11.4 predicts |
