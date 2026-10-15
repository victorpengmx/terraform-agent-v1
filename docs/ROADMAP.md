# Roadmap

**Current priority: the Resume MVP (M0–M7).** It's the smallest version that shows a self-built agent
runtime, measured. "After applying" waits until the MVP is done and applications are out. Every resume
claim must map to a file, a number or a demo. § references point to `docs/DESIGN.md`.

Sessions: name the item at the start; check it off and commit when done.

## Resume MVP

### M0. Setup

- [x] Repo skeleton per §19; `pyproject.toml` with `amulet-core==1.9.49` and `dev` / `render` extras;
      pytest running; `.gitignore` for `.venv`, `.env`, `traces/`, `renders/`, `worlds/` (**not** `docs/`)
      Differs from §19: the memory backend lives in `backends/base.py`, not `memory_backend.py`.
- [x] Amulet hello-world: write a 64×64 sine-wave hill into a closed flat 26.3 world; check in game
      (`HelloFlat_hello` exists; the Amulet backend tests run against it)
- [x] Verify 26.3 height limits in game; set `terrain/config.py` (result recorded there)
- [x] First commit; push to a (private for now) GitHub repo

### M1. Canvas and core ops, no LLM (§3, §4.1, §5)

- [x] `WorldState`, op registry (pydantic params → tool schemas, docstring → description), `apply_op`
      with validation and world-limit checks
- [x] Brush helpers (§5.4); shared `region` type (§5.2); `character` presets (§5.3)
- [x] Op stack: add/update/remove, seeded deterministic replay, stages, incremental apply, checkpoints;
      features derived from each op's diff (§4.1)
- [x] Terrain ops: `noise_base`, `raise_peak`, `raise_region`, `carve_valley`, `smooth`, `erode`, `warp`,
      `roughen`, `terrace` (§5.5)
- [x] Water ops: `carve_river`, `fill_lake`, `set_sea_level` with whole-block layout (§5.5)
- [x] Invariant tests per op, plus the no-raw-knobs schema test
- [ ] Calibration sweep script; verdict recorded in EVIDENCE.md
      Script done (`scripts/calibration_sweep.py`); no verdict in EVIDENCE.md yet.

### M2. Materials, compiler, backends, renders (§5.7, §7, §8, §10)

- [x] Palette rule engine (`banded`, `mix`, conditions), block list with looks, defaults;
      `set_palette_rules`, `paint_region`
- [x] Compiler: canvas → volume (cliff faces, subsurface, compile-time water containment) → diffed patch
- [x] Backend interface; memory backend; Amulet backend (copy-and-swap, snapshots, lock check, sky light)
- [x] L1 renderer; block colour table script; L2 renderer with camera presets
- [ ] Hand-scripted demos (valley, island, mesa) that look good in game; compile script with dry run
      Code done (`scripts/demos.py`, `scripts/compile_world.py --dry-run`); the in-game check is not
      recorded in EVIDENCE.md.
- [ ] Before/after images for the README (L2 + in-game)
      `scripts/readme_shots.py` exists; no `docs/images/` or README yet.

### M3. Harness v1 (§9, §13)

- [x] Model adapter (Anthropic) with price table; fake adapter for tests
- [x] Tools from the registry + `render_view`, `list_ops`, `update_op`, `remove_op`, `write_note`;
      feature stats in results
- [x] Inner loop with explicit stop-reason handling; budgets (steps, cost, time)
- [x] Context manager: episodes built from state, rotation, pinned scratchpad, byte-stable prefix
- [x] Tracing with run header; final op stack and renders saved (also on budget stop or crash)
- [x] Guardrail: max changed volume per call (§5.8)
- [x] Milestone: prompt → Haiku build → compiled world, checked in game
      (`traces/20261022-133743-e5a431.jsonl`, `TerraformTest` at x=900, z=0)

### M4. Critic and acceptance (§11)

- [x] Programmatic checks (§11.2 MVP list)
- [x] Critic: rubric, fresh context, strict JSON output, located issues with fix hints;
      `CriticError` → retry once, then record a failed round instead of crashing
- [x] Acceptance gate: objective rollback rules + blind comparison (§11.4), rollback feedback with the op
      diff, run ends on the latest accepted attempt (rule 2, the checklist pass count, comes with M5)
- [x] Gate validation: trace script for revision and position win rates; broken-copy test in both orders
- [x] Feature attributes v1 (§4.2): slope class, sinuosity, summit, prominence, reaches-edge; category
      functions shared with M5 and M6
- [x] (Optional) tool catalogue in the critic prompt (§11.5), with structured, schema-checked fixes

### M4.5 Tool quality pass (findings from the M3/M4 runs)

- [x] `carve_valley` cross-section: talus toe, rounded shoulder, steepness/roughness via `character`
- [x] Cliff palette: rock patches instead of bands by default, threshold 60°, soft slope limits
      (`slope_fuzz`) so moderate slopes are a grass/gravel/stone patchwork
- [ ] Optional: elongated rock streaks (veins across a face, scree chutes downhill) as a `mix` option
- [x] `meander` recalibrated so rivers can reach "twisty" and "meandering" (DESIGN §4.2 table)
- [ ] Seeded defaults with variation for omitted shape parameters
- [x] `region` polygons include their edges on every side (whole-area polygons missed east-edge columns)
- [x] Critic: materials in the rubric
- [ ] Critic: a second L2 angle (`overview_ne`) for critique and comparison
- [x] `raise_ridge` + spec extent (`span_x`/`span_y`) with checklist items (from the M5 image build)
- [ ] SPEC reads extent from photos too weakly: valley1.jpg's full-width range came out "wide", and its
      far-to-near flanks got no span (EVIDENCE 2026-10-25)
- [x] Critic: a separate `op_id` field for update_op/remove_op fixes

### M5. Spec and reference images (§9.3, §11.3, §12)

Design text and images together; build text first, then images on the same format.

- [x] Feature graph schema (features, types, categories, positions, v1 relations, `unsupported`) as a
      pydantic model; strict JSON output
- [x] SPEC step for text prompts; graph pinned in every episode and given to the critic
- [x] Optional `feature` argument on op tools; spec feature → op ids mapping; per-feature measurement (§4.3)
- [x] Code-checked checklist items with dependency skipping (§11.3); failed items fed back as located
      issues; pass count wired into the acceptance gate (rule 2, symmetric)
- [x] Reference images: SPEC from an image (+ optional text) with image-relative positions; code projection
      to (u, v) bands; critic sees the reference next to the L2 overview and fills `reference_match`
- [x] Milestone: one text and one image prompt built end to end and checked in game
      (`TerraformTest`: text at x=1200, z=300; image `valley1.jpg` at x=1200, z=600). The image build has the
      right features but single peaks where the photo has ranges (EVIDENCE 2026-10-23).

### M6. Mini-evaluation (§16)

- [ ] `eval/benchmark.yaml`: ~15 text + ~5 image prompts, each with a hand-written expected feature graph
      (ground truth for the checks)
- [ ] `eval/run_eval.py`: one row per run (success, judge score, checklist pass rate, steps, tokens, cost,
      latency, tokens/cost per successful build, model ids, git commit); text and image reported separately
- [ ] SPEC quality: agent graph vs. hand-written graph (features found, categories, relations)
- [ ] Eval judge with a prompt separate from the critic
- [ ] Ablations: critic on vs. off; SPEC on vs. off (2 seeds if budget allows)
- [ ] Report in `eval/reports/` with a results table and honest limitations

### M7. Presentable repo

- [ ] Make the repo public; confirm no keys in history
- [ ] README: what it is, architecture diagram, the LLM/code split in 3 sentences, a text and an image
      example (input → build), before/after images,
      results table, short related work (§17), limitations, next steps
- [ ] Interview prep: walk through the harness, critic, context manager and one op end to end; 2–3
      failures found in traces and how they were fixed (from EVIDENCE.md)
- [ ] Resume bullets from real numbers only

**If time is short**, cut in this order: the tool catalogue; the island/mesa demos; image input (ship
text-only, keep the design); v1 relations (keep features and categories, which the eval needs); the SPEC
on/off ablation. Never cut the acceptance gate, the checklist's category checks or the eval.

## After applying

Pick by benchmark failures, not by appeal.

### A. Spec, planning and stages (§9)
- [ ] CLARIFY before SPEC; user approval of the graph; richer relations (§9.3)
- [ ] Outer state machine with staged builds and objective stage gates (§9.1)
- [ ] Dominant-feature map, derived counts in observations, feature drift (§4.3)
- [ ] Mid-run user feedback merged into the spec

### B. Critic v2 (§11)
- [ ] VLM checklist items with an unverified state (§11.3)
- [ ] Annotated L1 renders + labels-vs-no-labels experiment (§10.4)
- [ ] More checks: large single-block areas, feature collisions, river connectivity, feature drift
- [ ] Second provider adapter; different-family critic and judge (§9.4)

### C. Tools (§5.10, §5.11, §6)
- [ ] Graded `carve_valley`; mountain-range recipe (`raise_ridge` done early, M4.5)
- [ ] Vanilla-style defaults; badlands palette preset (§6.3)
- [ ] Water hard cases: shallow fringes, river vs. lake, stepped-pool waterfalls
- [ ] `exclude` on `region`; minimum feature sizes (§6.2)
- [ ] `scatter_landforms` with relational placement; `flatten`, `tilt`
- [ ] `sculpt_3d` local voxel pass (needed for a Zhangjiajie showcase)
- [ ] `scatter_vegetation`
- [ ] Sandboxed `run_terrain_script` (§15)

### D. Reference images, precision (§12)
- [ ] Depth-ordering mini-experiment (Haiku vs. Opus vs. a depth model)
- [ ] `ref_view` camera matched to the photo; side-by-side critique from the same viewpoint
- [ ] Optional: depth model + segmentation; Set-of-Mark on references; sketch input

### E. Memory (§14)
- [ ] Recipe library seeded from reviewed hand-scripted builds; with/without ablation
- [ ] Promote recurring recipes to presets or macros; preferences (confirmed only)

### F. Full evaluation and baselines (§16)
- [ ] 30–50 prompts, more references, some sketches
- [ ] Human calibration (~40 ratings); pairwise/Elo judge; CLIP only if it tracks humans
- [ ] MCP server + generic-client baseline; optional VibeCraft comparison
- [ ] Ablations: low-level vs. semantic tools; model A vs. B; annotated renders; code-as-action

### G. Backends, renders and polish
- [ ] L3 capture script (§10.3)
- [ ] Live backend: GDPC (if 26.3 is supported) or a Paper plugin (§8)
- [ ] Demo video: prompt → staged build → critique fixing an issue → final flythrough
- [ ] Showcase build (e.g. Zhangjiajie karst pillars with foliage)
- [ ] Fine-tuning stretch: SFT on successful trajectories → DPO on judge-preferred pairs
