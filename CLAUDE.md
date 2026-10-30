# Terraform Agent

An LLM/VLM agent that turns text prompts and reference images (later also sketches) into large-scale
Minecraft terrain: mountains, valleys, rivers, mesas, karst pillars. The agent edits an in-memory **terrain
canvas** (layered heightmap + op stack + feature registry) through semantic tools, observes renders of its
work, is reviewed by a separate critic, revises, and the result is compiled into a Minecraft world.

Portfolio project for AI agent engineering roles. The point is the **agent runtime** (harness, tool
design, context management, critique, evaluation), not prompts on top of an API.

## Read first

- `docs/DESIGN.md`: the architecture. Source of truth for design decisions; §17 explains where ideas come
  from (related work), §18 lists fallbacks.
- `docs/ROADMAP.md`: build order and progress. **Current priority: the Resume MVP.**

## Guiding principles

1. **Code owns precision; the LLM owns intent and taste.** Tools take semantic parameters (normalized
   0–1 coordinates, named features, calibrated 0–1 scales, presets). Code does all geometry, physics rules
   and block placement. The LLM never emits blocks, raw grids or Minecraft commands.
2. **The canvas is the source of truth.** Renders are views of it; the Minecraft world is compiled from
   it through a swappable backend.
3. **Features, not pixels.** Specs, checks and critique talk about features (peak, ridge, river, mesa)
   with qualitative attributes and relations. Code measures what it can; the VLM judges only what code
   can't (looks, mood, naturalism).
4. **Minecraft quality is a constraint; fidelity is optimized within it.** A build is an interpretation of
   the prompt or reference, not a replica.
5. **Build bottom-up.** A tool must make good terrain when called from a human-written script before the
   agent relies on it.
6. **Everything is replayable.** Edits are ops in an op stack; undo and revision = edit the stack and
   replay.
7. **Evidence over intuition.** Every tool, prompt, critic or harness change records its evidence
   (traces, runs, renders) in `docs/EVIDENCE.md`, including reverted experiments. New machinery must show
   a benchmark or trace improvement.
8. **Fail loudly and informatively.** Errors go back to the agent as clear, fixable messages, never
   silently clamped.
9. **Simplest thing that works.** Add memory, extra model calls or new tools only when traces show the need.

## Tech stack

- Python 3.12
- numpy (amulet-core holds it at 1.26.x), scipy
- pydantic v2 (op parameter models = validation = LLM tool schemas)
- **Minecraft Java Edition 26.3** (pinned; don't upgrade mid-project)
- **amulet-core==1.9.49** (stable; first with 26.3 support). Don't use the 2.0.x alphas or any pre-release.
- matplotlib (L1 renders), PyVista offscreen (L2 voxel previews), Pillow
- anthropic SDK (first provider adapter), python-dotenv (`.env` holds `ANTHROPIC_API_KEY`; gitignored)
- google-genai (optional `[gemini]` extra; `.env` holds `GEMINI_API_KEY`), imported only inside `agent/gemini.py`
- pytest
- Later: official Python MCP SDK; Java 25+ and a stable Paper build for 26.3 (live backend)

## Models (DESIGN §9.4)

- Dev/debug runs: Claude Haiku 5.5 (cheap; use it while building plumbing)
- Builder for real runs: Claude Opus 5.5
- Critic: same adapter at first; a different model family later
- Gemini (any role): pass a `gemini-...` id to `--model` / `--critic-model`; the provider is inferred from the
  prefix and `make_adapter` builds the adapter. Gemini prices in `agent/models.py` are TODO (None): until
  they're filled in, cost is reported as "unknown" and the cost budget isn't enforced
- Eval judge: a prompt (later a model) separate from the in-loop critic
- Record exact model ids in every eval report.

## Conventions

- Ops are registered with a pydantic params model; the docstring becomes the tool description:
  ```python
  @register_op("carve_river", params=CarveRiverParams, stage="water")
  def carve_river(state: WorldState, p: CarveRiverParams) -> WorldState: ...
  ```
- Ops are pure: state in, new state out; no I/O, no globals. Callers go through `apply_op`, which
  validates, runs on a copy and checks world limits.
- Tool parameters follow DESIGN §5.1: shared `region` type for area-scoped ops, `character` presets
  (resolved values stored in the op stack), calibrated 0–1 scales, real units, defaults for all but the
  essentials, and **no raw algorithm knobs** (octaves, iterations, angles) in schemas. A test enforces it.
- Coordinates: normalized (u, v) in [0, 1]; (0, 0) = north-west corner; arrays are `[z, x]`; lengths are
  fractions of the area's shorter side. Conversion happens inside ops via the brush helpers.
- Heights are floats on the canvas; rounding to blocks happens only in the compiler.
- Every op has unit tests for its invariants (river bed never rises, lakes don't float, heights within
  world limits, no 1-column pillars).
- World height limits in `terrain/config.py` (`WORLD_MIN_Y = -64` inclusive, `WORLD_MAX_Y = 320`
  exclusive; verify in game in Phase 0). Never hard-code them.
- Tool lists and the system prompt are byte-stable within a run (prompt caching): deterministic order,
  no timestamps.
- Never commit Minecraft textures; only derived data (e.g. a block colour table).
- Never write to a world that may be open in the game; backends write to a copy and swap.
- Traces: one JSONL record per step in `traces/<run_id>.jsonl`; run artifacts in `renders/<run_id>/`.

## Commands

Fill in as they are created.

- Setup: `python -m venv .venv` then `.venv\Scripts\python -m pip install -e ".[dev,render,gemini]"`
  (`gemini` is optional: without it everything except the Gemini adapter works, and its contract tests skip)
- Tests: `pytest`
- Run agent: `python -m agent.run --prompt "..."`
- Gemini smoke check (paid): `python -m scripts.experiments.gemini_smoke --model gemini-...`
- Benchmark: `python -m eval.run_eval --suite eval/benchmark.yaml`

## Working agreement for sessions

- Start by naming the roadmap item. Use plan mode for anything touching more than one module.
- Keep `docs/DESIGN.md` in sync when a design decision changes; add an entry to `docs/EVIDENCE.md` for
  every tool, prompt, critic or harness change, including reverted experiments.
- When a tool schema changes, migrate saved op stacks and keep the demo scripts replaying.
- Check off the roadmap item and commit when it's done. Don't leave large uncommitted diffs between
  sessions.
