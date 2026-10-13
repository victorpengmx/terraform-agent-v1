# Terraform Agent — Design

Scope tags: **[MVP]** = part of the Resume MVP (see ROADMAP). **[Later]** = designed, built after the MVP.
Section numbers are referenced from CLAUDE.md and the roadmap; keep them stable.

## 1. Goal, scope and value

Turn a text prompt or reference image (later sketches) into a Minecraft landscape roughly 256–512 blocks
across. An agent plans, builds in stages, observes renders, is critiqued and revises; the result is
compiled into a Minecraft world.

- **In scope:** terrain, water, surface materials; later vegetation and local 3D features (overhangs,
  arches, caves, ledges).
- **Out of scope:** detailed structures (buildings). A `place_structure` tool (schematics) is possible later.
- **Why landscapes:** continuous, error-tolerant and mostly 2.5D. Structures are discrete, rule-bound,
  intolerant of small errors and truly 3D.

**Value vs. alternatives:**
- vs. a hand-written script per terrain: a script wins for one known terrain built once. The agent removes
  the programmer from each request, revises from feedback, checks itself, and reuses one op library.
- vs. an LLM writing scripts (code-as-action, e.g. SceneCraft): fixed tools enforce invariants (rivers
  descend, height limits), make edits replayable and revisable, cost fewer tokens, and give a stable
  interface that can be evaluated and improved.
- vs. block-level agents (Mineflayer bots, MCP servers with `place-block`): terrain scale is infeasible
  block by block; semantic ops make a full build ~20–60 tool calls.

## 2. The split: LLM vs. code

| LLM / VLM | Deterministic code |
|---|---|
| Interpret intent from text and reference images | All geometry: splines, noise, falloff, blending, erosion |
| Plan features and relations | Normalized → block coordinates |
| Choose tools, presets and semantic parameters | Physical rules (rivers descend, lakes fill to spill height) |
| Judge looks, mood, naturalism (critic) | Measure features; programmatic checks; guardrails |
| Decide when a stage looks done | Rendering, compiling, writing blocks |

**Tool abstraction is the key decision.** Too low (`setblock`): the model drowns in precise decisions it
is bad at. Too high (`make_volcanic_island()`): a template picker with no agency. Target: semantic ops with
rough parameters, e.g. `carve_river(path=[[0.1,0.8],[0.4,0.5],[0.9,0.3]], width=0.03, depth=4)`.
Validated later by an ablation (low-level vs. semantic tools, §16).

**Token efficiency** (cost grows with turns × context size):
- one semantic call replaces thousands of block edits; revise with `update_op` instead of stacking fixes
- system prompt + tool list byte-stable per run, so prompt caching works
- images dominate input cost: fixed sizes, only the latest renders in context, on-demand views capped
- programmatic checks before VLM calls; budgets on steps, cost and time
- report tokens and cost per successful build (§16)

## 3. The canvas [MVP]

The canvas is the in-memory working state: `WorldState` layers + op stack + feature registry. Renders are
views of it; the Minecraft world is compiled from it.

`WorldState` over W × D columns:
- `height`: float32 surface y per column
- `material`: int16 per column; 0 = "auto" (palette rules decide at compile time), other values index
  explicit overrides
- `water`: water surface y per column (NaN = dry)
- `masks`: named float masks (feature footprints `feature:<id>`; later protected areas, vegetation)
- `features`: registry of named features (§4)
- `voxel_regions`: bounded 3D boxes for local 3D edits [Later, §5.11]

Conventions: arrays shaped (depth, width), indexed `[z, x]`. (u, v) → x = u·(width−1), z = v·(depth−1);
(0, 0) is the north-west corner. Lengths are fractions of the shorter side, so circles stay circular on
non-square areas. Base: a flat area at a base y (then `noise_base` if wanted). Later: read existing terrain
from the world (+ a preserve mask for caves and builds), or seed from a reference image.

### 3.1 Op stack

```json
[
  {"id": "base",       "op": "noise_base",  "params": {"amplitude": 4, "character": "gentle"}},
  {"id": "mountain_1", "op": "raise_peak",  "params": {"center": [0.3,0.4], "radius": 0.2, "height": 90}},
  {"id": "river_1",    "op": "carve_river", "params": {"path": [[0.3,0.6],[0.6,0.7],[1.0,0.65]], "width": 0.03}},
  {"id": "erode_1",    "op": "erode",       "params": {"kind": "hydraulic", "strength": 0.5}}
]
```

- `replay(op_stack) -> WorldState` is deterministic. Each op's seed is derived from the run seed and its
  own id (e.g. `crc32(f"{base_seed}:{op_id}")`), so editing other ops never reshuffles its noise.
- **Incremental apply:** appending to the latest stage applies to the cached state; update, remove or an
  earlier-stage add replays. Every edit is all-or-nothing; errors name the op id.
- **Store resolved values:** presets (`character`) are expanded into the stored params, so replay never
  depends on the preset table; `update_op` re-derives values when a preset changes.
- Benefits: undo = remove an op; "taller mountain" = `update_op("mountain_1", {"height": 120})`; features
  are named; the stack is a compact text form for context, recipes and training data.

### 3.2 Stages

Replay order: `terrain → water → (compile) → voxel3d → materials → foliage`. Compile is a pipeline step,
not an op stage. Consequence: water ops always run after every terrain op, so erosion or smoothing can't
erase a river whatever order the model called them in. 3D and foliage ops anchor to named features so they
survive upstream edits.

### 3.3 Checkpoints

Immutable (op stack, cached state) snapshots. The harness snapshots each accepted attempt and restores on
rollback and at run end; per-stage snapshots come with multi-stage runs [Later].

## 4. Features

A feature is a named landform the agent and critic can talk about: the unit of planning, checking and
critique (principle 3).

### 4.1 Registry [MVP]

After each op, replay diffs height and water; the changed area becomes `features[op_id]` (normalized
bounds, top/bottom y, column count, water flag) plus a soft footprint mask. Ops don't need to know their
id. Tool results report the affected feature's stats; `list_ops` shows each op's footprint.

### 4.2 Types, measured attributes and categories [MVP v1, extended Later]

- **Type tag** per feature (peak, ridge, range, valley, river, lake, mesa, plateau, hills, pillar_field, …),
  taken from the spec feature an op realizes (§9.3), else inferred from the op.
- **Measured attributes** computed by code: summit y and prominence; mean/max slope and slope class; ridge
  orientation and length; river length, sinuosity, source and outlet, reaches-edge; plateau flatness;
  pillar count and sizes.
- **Categories:** the LLM states attributes as categories; code owns the numeric ranges and checks them.

| Attribute | Categories (code ranges) | Source |
|---|---|---|
| River winding (sinuosity = channel length / straight distance) | almost straight < 1.05; winding 1.05–1.25; twisty 1.25–1.50; meandering ≥ 1.50 | standard geomorphology classes ([Sinuosity](https://en.wikipedia.org/wiki/Sinuosity), after Mueller 1968) |
| Slope | gentle < 15°; moderate 15–35°; steep 35–60°; cliff > 60° | project-defined; calibrate by eye |
| Height | relative only: `taller_than`, `tallest`, `lower_than` | photos don't give absolute heights |

Checks allow one category of slack (one off = minor, two off = major); "unspecified" = no check. Calibrate
tool parameters against the categories (e.g. `meander` → sinuosity class) and record the table here.

**v1 measurement (`terrain/features.py`, M4):**
- Each form op is one feature: a peak, a valley, a river, a lake, a sea, or a `raise_region` plateau /
  basin / raised area / depression. Modifier and materials ops are not features.
- Attributes are measured over the feature's *body*, the columns where its footprint weight is > 0.5.
- **Slope class:** the mean slope over the body, on the lightly smoothed surface the palette uses.
- **Prominence:** classic topographic prominence from one union-find pass over the heightmap. A peak's
  prominence is its height above the highest col that connects it to higher ground; the area's highest
  point is measured against the area's lowest. A feature's prominence is the largest among the peaks in
  its body. Raised features only.
- **Sinuosity:** recomputed from the exact centre line `carve_river` cut (same params and seed).
- **Reaches-edge:** the area edges the footprint touches.
- Limitation: overlapping footprints share measurements. A whole-area seafloor reports the island on it,
  and island and volcano report one summit. The dominant-feature map (§4.3) fixes this later.

**`meander` → sinuosity (mean of 5 seeds; the same at 128 and 256 blocks):**

| meander | 0 | 0.2 | 0.4 | 0.6 | 0.8 | 1.0 |
|---|---|---|---|---|---|---|
| straight 2-point path | 1.00 | 1.01 | 1.02 | 1.05 | 1.08 | 1.12 |
| 3-point bent path | 1.08 | 1.08 | 1.10 | 1.12 | 1.16 | 1.20 |

`meander` alone never reaches "twisty" (1.25) or "meandering" (1.5); only path points bend a river that
far. The scale is miscalibrated against the categories (§5.9) and needs recalibrating before the M5
checklist asks for those classes.

### 4.3 Feature ↔ op mapping and overlaps [MVP v1 mapping; rest Later]

- One feature can come from several ops (mountain = `raise_peak` + `roughen` + `erode`), and one op can
  create many features (a scatter op → 40 pillars). [MVP] Every op tool takes an optional `feature`
  (a spec feature id); code keeps the spec feature → op ids mapping and measures each spec feature over
  the union of its ops' footprints. Ops without a feature are global or supporting edits.
- A per-column **dominant-feature label map** (largest height change wins) resolves overlaps for
  measurement, annotation (§10.4) and checks.
- **Counts are derived by code** and shown in observations ("3 peaks, 1 river, 1 pillar field (40
  pillars)"). The LLM never keeps counts; VLMs count poorly.
- **Feature drift warning:** if an op substantially changes an earlier feature's attributes (a plateau
  loses its flatness, a peak 30% of its height), the tool result warns. A warning, not a block.

## 5. Tools

Agent-facing tools = one tool per registered op (generated from the registry) + observe/control tools.

### 5.1 Tool design principles

1. **One tool = one designer intent.** General when builds differ by degree (gentle hills vs. jagged
   peaks); specialized when code must guarantee an invariant (water, non-overlapping scatter) or the
   intent has its own logic.
2. **No overlapping tools**: overlap causes inconsistent picks and wasted turns.
3. **Intent-level parameters only:** calibrated 0–1 scales (`roughness`, `steepness`, `strength`) and real
   units (blocks, normalized coordinates). Code derives algorithm knobs. A test asserts no raw knobs
   (octaves, iterations, angles, rates) appear in tool schemas. Models fill raw knobs with remembered
   constants, and critic hints like "lower sharpness to 0.2" fail validation on raw ranges.
4. **Defaults for everything but the essentials.**
5. **`character` presets** (§5.3): named bundles; explicit values override.
6. **One shared `region` type** for area-scoped tools (§5.2).
7. **Informative results and errors:** the affected feature's stats in every result; errors say what was
   wrong and by how much ("would raise terrain to y=341, above the build limit 319; reduce height by at
   least 22"). Ops that change only surface blocks say so ("surface blocks only, 2,310 columns set to
   gravel") so the model doesn't think nothing happened.
8. **Lean, byte-stable schemas** (prompt caching).

Grounded in [SWE-agent](https://arxiv.org/abs/2405.15793) (the agent–computer interface changes results a
lot at a fixed model) and Anthropic's [Writing effective tools for
agents](https://www.anthropic.com/engineering/writing-tools-for-agents).

### 5.2 Shared `region` [MVP; `exclude` Later]

```
region (omitted = whole area where allowed):
  {"circle":  {"center": [u, v], "radius": r}}
  {"polygon": [[u, v], ...]}                          # 3–32 points
  {"path":    {"points": [[u, v], ...], "width": w}}  # spline-smoothed through the points, slight width wobble
  {"feature": "mountain_1"}                           # an op's footprint
edge: fade width beyond the region (fraction of the shorter side)
```

One module builds a 0–1 weight from it on top of the brush helpers (§5.4). Polygons include their edges on every side, so
`[[0,0],[1,0],[1,1],[0,1]]` covers every column. Used by **modifiers**:
`raise_region`, `smooth`, `warp`, `roughen`, `erode`, `paint_region`. Ops whose geometry is their
definition keep their own parameters: `raise_peak` (center, radius), `carve_valley` and `carve_river`
(path). Paths must be smoothed: raw polylines read as zigzags with hard edges. All of them use one
Catmull-Rom curve through the points (`brushes.smooth_path`). [Later] `exclude: [region,
...]` keeps a precise feature out of a broad modifier; global application stays the default.

### 5.3 `character` presets and calibration [MVP]

| Preset | roughness | steepness |
|---|---|---|
| gentle | 0.20 | 0.25 |
| rolling | 0.35 | 0.40 |
| rugged (default) | 0.55 | 0.60 |
| jagged | 0.80 | 0.85 |

On `noise_base` (roughness only), `raise_peak`, `terrace`, `carve_valley`, later `raise_ridge`. Each op maps the scales
onto its internals (e.g. `raise_peak`: steepness → profile sharpness; roughness → outline irregularity and
radial gullies). **Calibration:** a sweep script renders each scale at 0.1 … 0.9; check by eye that 0.2
reads gentle and 0.8 jagged; record the verdict in EVIDENCE.md. Default noise toward gentle: fine noise on
gentle ground multiplies 1-block steps and reads as uneven.

### 5.4 Brush system [MVP]

Nearly every terrain op = **mask × falloff × shape × blend**: distances in blocks (circle, path →
distance + arc length, polygon → signed distance), weights from falloff curves (smoothstep, linear,
gaussian), combined by `blend(height, value, weight, mode)` with modes `add`, `max` (raise only), `min`
(carve only), `lerp_to_target` (flatten), `smooth` (blur). Ops never convert coordinates themselves.

### 5.5 MVP ops and their rules

**Terrain**
- `noise_base(amplitude, feature_size, character)`: gentle base relief.
- `raise_peak(center, radius, height, character | steepness, roughness)`: a peak. Must not read as a
  radially symmetric cone: domain-warped outline (lobes) and irregular radial gullies on mid-slopes, with
  the summit height preserved.
- `raise_region(region, height, mode: add | plateau | basin)`: raise or lower an area; negative height
  lowers; `basin` digs a flat floor below the outline's lowest point (always holds water).
- `carve_valley(path, width, depth, profile: v | u, character | steepness, roughness)`: river-cut V or glacial U
  (flat floor over 30% of the half width).
  - Walls are an S: a concave talus apron at the toe, the steepest point ~0.42 of the way up, and a broader
    convex shoulder that eases into the land at the rim. The old U (`1 − s⁴`) was steepest exactly at the
    rim and read as a ship hull.
  - steepness concentrates the face: max wall slope 1.6× (0) to 2.5× (1) the wall's mean slope.
  - roughness swells and narrows the width along the path (up to ±30%).
  - Depth relative to width sets how steep the walls are overall; the tool description says so.
- `smooth(region, strength)`.
- `erode(kind: thermal | hydraulic, strength, region?)`:
  - thermal: slopes crumble toward a talus angle; strength maps to passes and angle together
  - hydraulic: grid stream-power erosion (D8 routing, cut ∝ √(upstream area) × slope, transport capacity,
    deposition where slopes ease, mass-conserving). Route runoff on terrain + a small fixed noise so
    channels converge (on smooth shapes steepest-descent paths never merge). **Spread cuts and deposits
    over ~2 blocks**, or pits fill as single-column pillars. With a region: erode the whole area (runoff
    from outside counts), blend in by the region weight.
- `warp(region, amount, scale)`: moves terrain sideways along a smooth noise field, giving irregular
  outlines (scalloped mesa walls, ragged coasts, lobed peaks). Height noise can't move a wall; a sideways
  shift can. Fade ≥ 1.5× the shift.
- `roughen(region, amount, scale, style: fbm | ridged | billow, weight: uniform | relief | slope)`:
  vertical detail. **No noise features smaller than 4 blocks** (finer layers only make 1-column pillars).
  Noise is rescaled so `amount` is the true maximum. Ridged after erosion gives crags.
- `terrace(region, character)`: quantizes slopes into ledges and risers along contours (steepness → ledge
  vs. riser ratio and ledge tilt; roughness → step jitter; flats untouched). Layered cliffs = steep peak +
  warp + terrace + banded palette; terracing a creek's slope before carving gives stepped pools.

**Water** (invariants in code, not left to the model)
- `carve_river(path, width, depth, meander)`: the bed is the running minimum of (ground − depth) along the
  path, so it never rises and cuts gorges through ridges. The path must end at the area edge or existing
  water, else an error. **Lay water out in whole blocks** (bed floored, surface ≥ 1 block above, submerged
  columns a full block under the surface) and contain it against the banks *as the compiler will round
  them*, digging rather than draining. Otherwise shallow rivers round away at compile. The channel floor
  is half a path sample wide, so columns between samples never sit above the bed.
- `fill_lake(at, max_depth)`: priority flood from the border gives each column's spill level; fill the
  connected basin to min(spill, ground + max_depth). No basin → error.
- `set_sea_level(level)`.
- Known hard cases to design for: shallow lake fringes rounding away; a river carved after a lake draining
  it; waterfalls (static per-column water can't fall; use stepped pools).

**Materials** (§5.7): `set_palette_rules`, `paint_region`.

### 5.6 Observe and control tools [MVP]

- `render_view(view, feature?)`: on-demand 512 px L1 map or L2 corner view, optional feature close-up;
  capped per run; no repeat renders of unchanged terrain. Without it the builder builds blind until the
  next round.
- `list_ops`, `update_op(id, params)`, `remove_op(id)`, `write_note(note)` (scratchpad, §13).
- Checkpoint and rollback belong to the harness, not exposed as tools.

### 5.7 Materials [MVP]

Rule engine per surface cell, first match wins. **Rules are structured data, not expressions** (safe,
validated, clean schema):

```json
{"rules": [
  {"when": {"slope_min": 60}, "use": {"kind": "banded", "blocks": ["stone","andesite","tuff"], "band_min": 2, "band_max": 5, "noise": 0.3}},
  {"when": {"water_dist_max": 3}, "use": {"kind": "mix", "blocks": {"gravel": 0.5, "mud": 0.3, "sand": 0.2}}},
  {"use": {"kind": "mix", "blocks": {"grass_block": 0.8, "podzol": 0.2}}, "subsurface": "dirt"}
]}
```

- Conditions: `slope_min/max` (degrees, measured on a lightly smoothed surface so rock follows landforms
  instead of speckling grassland at 1-block steps), `y_min/max`, `water_dist_max`, `feature`, and
  `slope_fuzz` (degrees): the slope limits wander by up to that much in large coherent patches. Ground near
  a limit is then partly one rule and partly the next, so a 30° slope can be grass in one place and rock in
  another, instead of everything switching at one angle.
- Blocks from a closed list of natural blocks, **each described by how it looks** ("granite: dull pinkish
  brown"), test-checked against a measured colour table. Models otherwise pick blocks from memory and get
  colours wrong.
- `banded`: block by y + noise → strata; noise also tilts the strata slightly so bands on a cone don't
  form concentric rings. `mix`: weighted random with spatial noise → patches, not static.
  The tilt is not enough on steep cones: default bands still read as rings (EVIDENCE 2026-10-22), so the
  default palette has no bands. Keep `banded` for strata the prompt asks for (mesas, badlands).
- Defaults exist; the model only overrides. `paint_region(region, block)` overrides rules locally.
  The default palette (M4.5) has:
  - cliffs: from 60° ± 10°, patches of stone, andesite, tuff and gravel
  - snow: above y 170
  - shores: within 2 blocks of water
  - moderate slopes: from 35° ± 12°, large patches of grass, coarse dirt, stone, andesite and gravel
  - grass: everywhere else

  Rock share by slope on constant-slope ramps: 20° 0%, 30° 15%, 40° 34%, 50° 42%, 60° 68%, 70°+ 100%.
  Before this change it jumped at 35° and 50°.
- [Later] `apply_palette_preset(style)` (temperate, alpine, badlands, karst) if traces show long or invalid
  rule calls.

### 5.8 Guardrails

- [MVP] schema validation (unknown fields rejected), world height limits in `apply_op`, all-or-nothing
  edits, agent-readable errors with fix amounts.
- [MVP] max changed volume per call (sum |Δh|, configurable); the error suggests narrowing the region or
  lowering strength. Limit: 40 blocks per column on average (`MAX_MEAN_DH`), checked on agent `add` and
  `update_op`, not on `remove_op` (undo is never limited). The largest demo edit is 26.
- [Later] protected masks (user-drawn), feature drift warnings (§4.3).

### 5.9 Trace-driven tool review

| Signal in traces | Likely cause | Action |
|---|---|---|
| Wrong or unknown tool picked | overlap or unclear description | sharpen description, remove overlap |
| Repeated errors on one tool | unclear schema or message | fix schema or message |
| A parameter pinned at its extreme | miscalibrated scale | recalibrate |
| Same multi-call sequence every run | missing abstraction | preset, macro or recipe |
| Fixes stacked instead of `update_op` | revision path not obvious | prompt or description |
| Critic hints naming non-existent params | critic doesn't know the tools | tool catalogue for the critic (§11.5) |

### 5.10 Later ops

- `raise_ridge(path, height, character)` + a graded `carve_valley` (floor descends to its exit): a mountain
  range becomes ridges + spurs + saddles, not one cone.
- `scatter_landforms(kind, region, count, size_range, height_range, min_spacing, distribution, character)`:
  pillar fields, hill clusters, boulders; Poisson-disk spacing; placement from stated relations, solved by
  code (Holodeck-style, §17).
- `flatten` (or document `raise_region(mode="plateau")`), `tilt` (ramp), `add_waterfall`.
- `scatter_vegetation(region, species, density)`: from masks (slope, material, ledges), Poisson-disk
  spacing, procedural trees or a small schematic set; vines on wall faces.
- `run_terrain_script` (code-as-action escape hatch, §15).

### 5.11 Local 3D pass [Later]

The heightmap can't represent solid above air (arches, overhangs, caves). One tool,
`sculpt_3d(kind: overhang | arch | cave | ledge, anchor, size, character)`: after compiling the 2.5D
terrain, cut a bounded voxel box (e.g. 64³) around the anchor feature, apply SDF-style add/subtract, write
it back. Runs in the `voxel3d` stage; needs angled views (L2/L3) to judge.

## 6. Minecraft quality and fidelity

### 6.1 Priority

Minecraft quality is a **constraint**; fidelity to the prompt or reference is optimized within it. A
needle-thin spire from a photo becomes the thinnest spire that still reads well. The feature spec (§9.3)
is the fidelity contract: what it names must match; everything else is free, which is where
Minecraft-friendly choices happen. The critic judges an interpretation, not a replica.

### 6.2 Block-scale rules

| Problem | Cause | Rule |
|---|---|---|
| 1-column pillars and spikes | fine noise, erosion pits | ≥ 4-block noise features; spread deposits; `no_spikes` check [MVP] |
| Thin features read as noise | pillars or ridges a few blocks wide | minimum sizes in validation, e.g. pillar radius ≥ 4–5 blocks, river ≥ 3 blocks wide [Later] |
| "Contour line" terraces on gentle slopes | slight slopes → long single-block steps | gentle default noise; small roughen; texture mixes; check the share of 1-block steps [MVP check] |
| Stair-stepped mid slopes | 1-block steps at ~25–45° | palette rules: stone/gravel mixes hide steps better than grass [MVP] |
| Real proportions don't fit | build height y −64..319; subtle relief looks flat in blocks | scale translation: compress very tall features, exaggerate subtle relief [guidance] |
| Flat colour areas | one block over a large area | `mix` / `banded` palettes [MVP]; check large single-block areas [Later] |
| Water above banks or vanishing at compile | rounding | whole-block water layout and compile-time containment (§5.5, §7) [MVP] |

### 6.3 Vanilla-style defaults [Later; pull in if evals show the need]

When the prompt is vague, default to terrain that looks like vanilla Minecraft: mountains as ranges with
ridgelines, spurs, saddles and valleys (not single cones); badlands with banded terracotta, irregular flat
tops, scree aprons and sand plains; plains with gentle relief and mixed grass. Cheap models otherwise
produce smooth blobs, one-block palettes and cone mountains.

## 7. Compiler [MVP]

Deterministic, no LLM. Compiles the canvas to a dense block volume (`[z, y, x]` + palette); heights are
rounded only here. Per column:

1. surface block from the material rules at `h`, extended down any exposed cliff face (height above the
   lowest 4-neighbour) so walls show banded rock, not soil; the rule's subsurface layers below (default 3);
   stone underneath; air above
2. water up to the water level where `h < water`; after rounding, re-contain water (lower, never raise): no
   water block beside lower dry ground, never air under water
3. [Later] voxel regions; foliage

`build_patch(state, backend, origin)` compiles over a y range covering both the new terrain and the
world's current surface, then diffs against `backend.read_region`: a patch of changed blocks only, which
doubles as the undo record. Re-running on an applied world gives an empty patch.

## 8. Backends

```python
class WorldBackend(ABC):
    def read_region(self, box) -> Volume: ...
    def apply_patch(self, patch) -> None: ...
    def snapshot(self) -> str: ...
    def restore(self, snapshot_id: str) -> None: ...
```

- **Amulet backend [MVP, primary]:** offline, headless, repeatable; used for development and evals.
  - Refuse a world whose `session.lock` is held (the game has it open).
  - Write to a copy, save, then swap; keep the old world as a snapshot outside `saves/` on the same drive
    (swaps are renames).
  - **Lighting:** amulet-core's save drops light data, and 26.3 may relight only some sections (dark
    terrain above ~y 95). Write vertical sky light for changed chunks (2.5D: 15 in clear blocks, −1 per
    water/ice/leaves block, 0 in and below opaque blocks), zero block light, `isLightOn=1`.
  - Missing chunks → a clear error ("fly over the area first").
  - 26.x stores overworld chunks under `dimensions/minecraft/overworld/region/`; amulet-core 1.9.49
    handles it.
- **Memory backend [MVP]:** a superflat world in memory for compile → diff → apply → undo tests.
- **Live backend [Later, demo only]:** a small Paper plugin (HTTP/WebSocket, batched and throttled writes,
  physics suppressed, diff-based undo, resumable batches), or GDPC + the GDMC HTTP Interface mod if it
  supports 26.3. Same interface; the agent doesn't change.
- **Rejected as primary:** Mineflayer-style bots (too slow at terrain scale, no physics control, protocol
  lag); RCON `/fill` (32,768 blocks per command, poor feedback); WorldEdit commands (stateful selections;
  brushes need a player; weak read-back; no erosion or rivers); Axiom (its tools run client-side; humans
  may co-edit and the agent re-reads the region).
- Optional later: a bot as an in-game avatar/chat front end only; edits still go through a backend.

## 9. Harness (agent runtime)

### 9.1 Outer loop

- [MVP] SPEC → one `terrain` stage of BUILD → CRITIQUE → (REVISE | ACCEPT) for up to `max_rounds`. SPEC
  turns the prompt and/or reference image into the feature graph (§9.3), which is pinned for the run. The
  run ends on the latest accepted attempt (§11.4).
- [Later] `CLARIFY → PLAN/SPEC → per stage [BUILD → OBSERVE → CRITIQUE → (REVISE | ACCEPT)] → COMMIT`.
  - CLARIFY: ask only questions whose wrong guess is expensive (scale, area, count).
  - PLAN/SPEC: the feature graph (§9.3), optionally shown to the user for approval.
  - **Stage gates on objective conditions, not model confidence** (verbalized LLM confidence is poorly
    calibrated, [Xiong et al.](https://arxiv.org/abs/2306.13063)): blocking checks pass; code-checked spec
    items pass or are marked unfixable; the comparison gate doesn't prefer the previous attempt. Otherwise
    stop at the round limit and record why.
  - Mid-run user feedback is merged into the spec; the run continues from current state.

### 9.2 Inner tool loop [MVP]

Model call → execute tool calls → feed back results or errors, until the model ends its turn, a budget
stops it or it refuses. Handle every stop reason explicitly: `tool_use`; `end_turn`; `max_tokens`
(truncated tool calls are not run; ask for a re-issue); `pause_turn` (resend); `refusal`. Budgets: model
calls (builder + critic), cost, wall time; critique rounds. Agent-fixable problems come back as error tool
results; any other exception is a bug and propagates.

### 9.3 Plan and spec: the feature graph [MVP v1; extensions Later]

A SPEC step writes a small graph before building. After SceneCraft's relational scene
graph and 3D-GPT's prompt expansion (§17):

```json
{
  "features": [
    {"id": "range_1", "type": "mountain_range", "position": "back", "orientation": "crosswise", "height": "tall", "character": "jagged"},
    {"id": "peak_1",  "type": "peak", "part_of": "range_1", "position": "left", "height": "tallest"},
    {"id": "hills_1", "type": "hills", "position": "right", "character": "rolling"},
    {"id": "river_1", "type": "river", "winding": "winding"}
  ],
  "relations": [
    ["river_1", "flows_from", "range_1"], ["river_1", "reaches", "south_edge"],
    ["hills_1", "lower_than", "range_1"], ["peak_1", "left_of", "hills_1"]
  ],
  "materials": "grey banded rock on cliffs, grass and podzol below",
  "unsupported": ["mist"]
}
```

- Attributes are categories (§4.2); positions are qualitative. Keep it small (5–15 features).
- Relations: positional (`left_of`, `behind`), comparative (`taller_than`), topological (`flows_from`,
  `reaches`, `surrounds`, `part_of`); topological ones need their own checks.
- Uses: pinned spec; builder plan; critic checklist (§11.3); eval checks (§16); recipe description (§14);
  target format for reference images (§12).

**MVP v1:**
- A SPEC model call (builder model, fresh context, strict JSON schema) turns the text prompt and/or
  reference image into the graph. Features, types, categories (§4.2), qualitative positions and a few
  relations. Image positions are image-relative and projected by code (§12).
- Positions: `left | center | right` × `back | middle | front` (text), or image-relative terms
  (images). Code maps them to (u, v) bands for checks; the builder sees both.
- Supported relations in v1: `left_of`, `right_of`, `behind`, `in_front_of`, `taller_than`,
  `lower_than`, `tallest`, `flows_from`, `reaches` (an edge or a feature), `part_of`.
- The graph is pinned in every episode and given to the critic. The builder names the spec feature each op
  realizes (§4.3).
- Unsupported requests (mist, buildings, vegetation before it exists) go to `unsupported`, so neither the
  builder nor the critic chases them.

**Later:** CLARIFY before SPEC; user approval of the graph; graph edits from mid-run feedback; richer
relations (`surrounds`, `between`, `along`).

### 9.4 Model adapter and roles

- [MVP] One adapter interface for every role; Anthropic first. Messages and tools may use the Anthropic
  format internally; other providers translate at the adapter boundary. A price table drives budgets and
  reports.
- [Later] Portability: provider SDKs only inside adapters; capability flags (images, caching, parallel
  tools, JSON-schema output), not provider checks; normalized usage and cost; retries and rate limits in
  the adapter; contract tests with recorded responses.

| Role | Needs | Choice |
|---|---|---|
| Builder | tool use, long multi-turn reasoning, vision | Claude Opus 5.5 |
| Dev/debug | cheap and fast | Claude Haiku 5.5 |
| Critic | visual judgment, structured output | same adapter first; different family later (less self-bias) |
| Eval judge | independent of the critic | separate prompt first; separate model later |
| Fine-tuning (stretch) | open weights | small open VLM (Qwen3.5 / Gemma 4 class) |

Model choices are decided by the benchmark and human calibration, not public leaderboards.

### 9.5 Tracing [MVP]

One JSONL record per step: run id, stage, step, model, request id, stop reason, tokens (incl. cache),
cost, latency, assistant text, tool calls (name, input, result, error flag), critic output, and a note
(rollbacks, rotations, run end). Step 0 holds a header: prompt, reference image paths + hashes, models,
effort, rounds, render level, budget, canvas, git commit + dirty flag, a hash of the tool list. A run can be
reproduced or re-scored from its trace. Save the final op stack and renders to `renders/<run_id>/`, also on
a budget stop or crash.

## 10. Observation: the fidelity ladder

| Level | Content | Cost | When |
|---|---|---|---|
| L1 [MVP] | top-down map: hillshade, land tint, water depth, contours, 0.1 (u, v) grid, north arrow, scale bar | ms | round start, critique, `render_view(map)` |
| L2 [MVP] | voxel preview of the compiled blocks | seconds | round start, critique and blind comparison (default; `--render l1` to skip), `render_view(3d)`, final save |
| L3 [Later] | real Minecraft render | slow | milestones, final critique |

Text alongside images: terrain stats, condensed op list, per-feature stats (later counts and measured
attributes, §4). **The model never reads raw arrays.** Every image carries a compass and scale so critic
issues map back to coordinates; image sizes are fixed (1024 px; 512 px on demand) to control cost.

### 10.1 L1

Matplotlib, fast. Good for layout and the derived plan ("does the river run south-west to north-east?"),
not for comparing with perspective photos.

### 10.2 L2

PyVista offscreen. Visible faces only (a face only where a block touches air or water); per-face baked
shading (fixed brightness per direction + top-face ambient occlusion from taller neighbours) for a crisp
block look; translucent water. Camera presets `overview_ne/sw/se/nw` and `close_up(feature)` from a
feature's bounds; later `ref_view` matched to a reference photo (§12). Colours from a script that reads the
local 26.3 client jar, follows block-model parents, averages opaque texture pixels and applies plains
tints, then writes a colour table (numbers only, no textures). Greedy meshing only if speed requires.

### 10.3 L3 [Later]

In-game screenshots from a real 26.3 client: a capture script opens the committed world, teleports to the
preset poses, triggers screenshots and collects the newest files. Always matches the game version. Without
a client (unattended evals), fall back to L2. Offline path tracers (e.g. Chunky) only if verified on 26.3.

### 10.4 Annotated renders [Later]

[Set-of-Mark prompting](https://arxiv.org/abs/2310.11441) (numeric labels on segmented regions) raised
GPT-4V's referring-expression accuracy on RefCOCOg from 25.7 to 86.4. We control our renders, so:
- an **annotated L1**: outlines from the dominant-feature map (§4.3) plus feature-id labels at each
  region's most interior point; the normal L2 stays unannotated for judging looks
- never repaint the terrain: annotation is an overlay
- labels over colours (terrain has its own greens and greys); crowded scenes label groups and use
  `close_up(feature)`
- validate with a small experiment (does the critic name the right op id more often with labels?); the
  published gain is on photos, not blocky renders

## 11. Critic

### 11.1 Base design [MVP]

- Inference-time self-refinement (Self-Refine / Reflexion style); no weights change.
- **Fresh context:** never sees the builder's conversation. Inputs: spec, references, renders, terrain
  stats, condensed op list, programmatic check results.
- **Programmatic checks first** (cheap, objective); blocking failures skip the model call.
- Rubric 1–5 with level descriptions: prompt_adherence, layout, naturalism, minecraft_quality,
  reference_match (null without references). Structured output (strict JSON schema), issues before scores
  so the model reasons before rating. Each issue is located (op id or u,v) with a concrete fix hint naming
  a tool and parameter. Verdict "pass" only with no major issues and all scores ≥ 4.
- The critic judges and suggests; the builder decides and acts. Accept/rollback lives in the harness.

Grounding matters: unaided self-correction is unreliable ([Huang et
al.](https://arxiv.org/abs/2310.01798)); tool-grounded critique works much better
([CRITIC](https://arxiv.org/abs/2305.11738)).

### 11.2 Programmatic checks

- [MVP] built something; no spikes (a column far above all 4 neighbours); slope sanity (share of columns
  above ~70°, share of 1-block steps on gentle ground); water containment; meaningful change vs. the flat
  base; thin features.

  Implemented in M4 (`agent/critic.py` `run_checks`). Thresholds are set so the three demos and the
  baseline run pass and constructed broken terrains fail (EVIDENCE 2026-10-22):

  | Check | Severity | Measure | Fails when |
  |---|---|---|---|
  | built_something | blocking | ops applied | none |
  | meaningful_change | blocking | columns ≥ 1 block off the flat base, or wet | < 1% (demos: 57–98%) |
  | no_spikes | major | columns > 3 blocks above all 4 neighbours | any |
  | steep_share | major | columns dropping ≥ 3 blocks (≈ 72°) to a neighbour, on rounded heights | > 25% (demos: 0.2–10.4%) |
  | gentle_steps | info | isolated 1-block bumps or pits on gentle ground | never; reported only |
  | water_containment | major | canvas-wet columns that are dry after compile rounding | > 25% (demos: 1–9%, lake fringes) |
  | thin_features | major | rounded height ≥ 4 above its 3×3 grey opening (1–2 block pillars and walls) | > 0.2% of columns (demos: ≤ 0.05%) |

  Slope is per block, not a central gradient: central differences compare a column's two neighbours and
  miss walls that alternate every column. Every message gives the measured number and an example (u, v).
- [Later] large single-block areas; feature collisions; river connectivity; heights vs. spec; feature drift.

### 11.3 Spec checklist [MVP: code-checked items; Later: VLM items]

Turn the feature graph into atomic items, in the style of [TIFA](https://arxiv.org/abs/2303.11897) and the
[Davidsonian Scene Graph](https://arxiv.org/abs/2310.18235) (atomic, non-overlapping questions with
dependencies) and [CADCodeVerify](https://arxiv.org/abs/2410.05340) (verification questions answered on
renders):

| Item | Judge |
|---|---|
| river_1 exists; flows from range_1; reaches the south edge | code |
| peak_1 is the tallest and left of hills_1 (in the reference's frame, §12) | code |
| river_1 is "winding" (sinuosity class) | code |
| hills_1 "rolling" = gentle/moderate slope class | code |
| looks natural; matches mood and palette | VLM, on L2 (and `ref_view`) |

- **Dependencies:** if "river_1 exists" fails, skip its child items. [MVP]
- **Code-checked items** (existence, categories, relative position and height, `flows_from`, `reaches`)
  are computed from measured feature attributes (§4.2) and the mapping (§4.3). [MVP]
- **VLM items** (looks, mood, palette) as yes/no questions. When the VLM is unsure, mark the item
  unverified, not failed, so noise doesn't trigger rollbacks. [Later; the rubric covers looks in the MVP]
- Failed items go to the builder as located issues ("river_1: sinuosity 1.02 (almost straight), spec
  says winding; raise `meander`").
- The pass rate is the acceptance gate's rule 2 (§11.4) and an eval metric (constraint passing, as in
  SceneCraft).

### 11.4 Acceptance gate [MVP]

Do **not** gate on absolute rubric scores: a fresh critic call scores near-identical terrains ±0.2 apart
and ties often, so a strict "higher mean score" gate throws away real improvements. Do **not** use an
in-call comparison labelled "previous"/"revision": models favour the item framed as the improvement and
keep the revision almost always. Design:

1. **Roll back** if the revision fails a blocking or major programmatic check the previous attempt passed.
2. **Roll back** if the spec checklist pass count drops (§11.3).
3. Otherwise a **blind comparison**, a call separate from the critique:
   - the two attempts shown as "Terrain 1" and "Terrain 2" in a seeded random order (seed = run seed +
     round); nothing says which is newer; record the order in the trace
   - inputs: spec, references, each terrain's renders and stats; **no** op lists (ids reveal recency),
     **no** previous critique
   - neutral question ("which better fulfils the spec … 'same' if neither is clearly better"); symmetric
     output: `strengths_1`, `strengths_2`, `better: "1" | "2" | "same"`, `why`
   - roll back only if the previous attempt wins; ties keep the revision
   - `--compare-both-orders`: judge both orders and accept a winner only if both agree (2× cost; for Opus
     and eval runs)
4. **Rollback feedback:** before restoring, compute the op diff (added, removed, changed params old → new)
   and give the next episode the diff, `why` and the rejected attempt's issues.
5. The run ends on the latest accepted attempt.

Implementation details (M4, `agent/harness.py`):
- **Baseline.** Round 1's attempt is accepted as the baseline, failed checks or not: there is nothing to
  compare it with. A "pass" verdict ends the run only on an accepted attempt.
- **Rule 2** arrives with the spec checklist (M5); until then, rule 1 goes straight to rule 3.
- **Order.** `random.Random(run_seed * 1000 + round)` decides whether the revision is Terrain 1.
- **Neutral renders.** Comparison renders use the renderers' neutral titles (size and height range);
  builder and critic renders are titled with the op count, which would reveal recency.
- **Unusable replies.**
  - Critique: retried once. A second failure records a *failed round*: checks only, `failed` set,
    verdict "revise", traced. The run continues.
  - Comparison: retried once, then counts as no winner, so the revision is kept, like a tie. Both
    failures are traced.
- **Cost.** A failed call is still charged to the budget (`CriticError` carries the response).

Validation: revision win rate clearly below 100% (above ~90% over 20+ comparisons = still biased); "Terrain
1" win rate near 50% (else enable both orders); a constructed test (a terrain vs. a broken copy, both
orders) must pick the intact one. Tools: `scripts/gate_stats.py` (win rates from traces) and
`scripts/gate_check.py` (broken copy = the valley demo without water plus a heavy fine ridged roughen).

### 11.5 Other critic features

- [MVP, optional] **Tool catalogue** in the critic's system prompt (one line per op: name, first sentence,
  parameter names; generated from the registry; cached). Fix hints then name real tools; count "no tool"
  hints as a ranked list of missing tools.
  Implemented in M4 (`agent/critic.py`):
  - **Catalogue** (`tool_catalogue`): every parameter is shown with its allowed values (`profile=v|u`) or
    range (`width [0..1]`), plus `update_op`/`remove_op`, the character presets and the region shapes.
  - **Structured fix:** each issue states its fix as `tool` (an op, `update_op`, `remove_op` or `"none"`) and
    `changes` (param, JSON value), next to the free-text `fix_hint`. `update_op`/`remove_op`
    fixes name their target in `op_id`; `location` stays an op id or a u,v position.
  - **Validation:** `hint_problems` checks a fix against the schemas (tool, op id, parameter, value).
  - **Builder:** the builder sees the structured fix after the hint.
  - **Missing tools:** `scripts/missing_tools.py` ranks `"none"` issues across traces.
  - **A/B:** `scripts/catalogue_ab.py` re-critiques saved runs with and without the catalogue (EVIDENCE
    2026-10-23).
- [Later] Annotated L1 (§10.4) so issues are located by feature id.
- [Later] A different-family critic once a second adapter exists.

## 12. Reference images [MVP minimal; precision Later]

VLMs order depth reasonably well (occlusion, haze, size, position in frame) but are poor at metric
distance and at projecting a perspective photo into a top-down layout; small models are weaker.
[MineAnyBuild](https://arxiv.org/abs/2505.20148) benchmarks this spatial-planning weakness in Minecraft
building. Design:

1. **Photo → feature graph** (frontier VLM): features, categories, relations, and positions in the image's
   own terms (`image_x: "center-right", depth: "middle"`), not map coordinates.
2. **Code projects** to (u, v) with a fixed camera assumption: left–right → u; near/middle/far → v bands;
   the frame widens with distance.
3. Optional precision: a monocular depth model (e.g. Depth Anything) + rough segmentation projects
   features onto the ground plane; can also seed the canvas (`image_to_heightmap`).
4. Optional: Set-of-Mark on the reference itself (segment with SAM, label regions) so the VLM builds the
   graph by pointing at regions.
5. **Compare like with like:** an L2 render from `ref_view` next to the photo. Never ask a VLM to compare a
   top-down map with a perspective photo.
6. Photo slopes: categories and relative constraints only. Exact angles (e.g. ±5°) are ill-posed under
   foreshortening; the slope perpendicular to the view can't be inferred, so leave it to defaults and
   erosion.
7. **Sketch maps** (top-down drawings) remove the projection problem: photos for mood, sketches for layout.

**MVP (minimal):** steps 1, 2 and the critic part of 5 with the standard L2 overview instead of
`ref_view`: the critic sees the reference next to the L2 overview and fills `reference_match`. Images go
through the same SPEC step and feature graph as text, so building, checks and eval are shared. Text and
image can be combined ("like this photo, but with a lake").

**Later:** `ref_view` camera matched to the photo; depth model and segmentation (step 3); Set-of-Mark on
references (step 4); sketch maps (step 7). Run a cheap experiment first: 15–20 photos with hand-labelled
depth-ordering questions; compare Haiku, Opus and a depth model.

## 13. Context management [MVP]

- Each turn the builder sees: system prompt + tool schemas (fixed per run, cached), pinned spec, terrain
  stats and op list, last critique (or rollback feedback), scratchpad notes, latest renders.
- **Episodes, not history edits.** Current Claude models bind thinking blocks to the conversation, so
  rewriting earlier turns (dropping renders, summarizing) is invalid. Each BUILD round opens a fresh
  episode whose first message is generated from state; within it history is append-only. When an episode
  gets long (input tokens or message count), rotate to a new one built from current state. Compaction is
  cheap because the op list, not the chat, is the source of truth.
- `render_view` images are tool results, so they stay until the episode ends; cap them per run.
- Scratchpad: a `write_note` tool; notes are pinned into every episode.

## 14. Memory [Later]

| Store | Purpose | Format | Retrieval |
|---|---|---|---|
| Canvas + op stack | session working memory | in memory; op stack JSON | direct [MVP] |
| Scratchpad | notes that survive episode rotation | list | pinned [MVP] |
| Recipe library | reusable successful builds | SQLite or JSON + embeddings | tags + cosine top-k |
| Promoted skills | generalized recipes | presets, macros or tools | called directly |
| Preferences | cross-session style | small JSON | loaded at start |
| Traces | debugging, evals, training data | JSONL | offline [MVP] |

Two levels, after SceneCraft's **non-parametric** library learning (no fine-tuning; a library improved
across runs):
- **Recipes:** `{description, prompt, spec, op_stack, scores, checks_passed, tags, tool_schema_version}`.
  Admit only runs that passed the critic above a threshold; dedupe. Retrieve by tags then cosine similarity
  of text embeddings (in-memory numpy is enough at hundreds of recipes; no vector DB). Inject 1–3
  condensed recipes (spec + op stack) at PLAN time. First recipes: reviewed hand-scripted builds (mountain
  range, cliffed horn + terraced creek, badlands mesa).
- **Promoted skills:** recurring recipes become presets, macros or tools.
- Version recipes with the tool schema; migrate or drop on changes.
- Must beat a with/without benchmark comparison.
- Preferences: save inferred ones only after the user confirms.

## 15. Code-as-action escape hatch [Later]

`run_terrain_script(code)`: the agent writes a small `edit(height, mask, ...) -> arrays` for requests no
tool expresses. Sandbox: separate process, numpy only, no file or network access, time and memory limits,
works on copies; output validated (shape, ranges, only inside the allowed region). The prompt prefers
tools. Log every script; recurring ones become tools.

## 16. Evaluation

**[MVP]**
- `eval/benchmark.yaml`: 15–20 prompts: ~15 text and ~5 reference images (some with extra text). Each
  prompt has a **hand-written expected feature graph** (the ground truth) whose code-checked items are the
  eval checks. Builds are checked against the hand-written graph, not the agent's own SPEC output;
  otherwise image prompts would only measure internal consistency. SPEC quality is reported separately:
  how well the agent's graph matches the hand-written one (features found, categories, relations).
- `eval/run_eval.py`: one row per run: success (checks pass and judge passes), judge score, **checklist
  pass rate**, steps, tokens, cost, latency, **tokens and cost per successful build**, model ids, git
  commit; text and image prompts reported separately.
- **Eval judge** with a prompt separate from the in-loop critic, to avoid optimizing against the measure.
- **Ablations:** critic on vs. off; SPEC on vs. off (builder gets the raw prompt or image; checks still
  use the hand-written graph). Same prompts and models; 2 seeds if budget allows.
- Report in `eval/reports/` with honest limitations (small N, judge not human-calibrated).

**[Later]**
- 30–50 prompts, more references, some sketches.
- Human calibration: ~40 ratings incl. "faithful but ugly" and "pretty but loose" cases; judge–human
  agreement.
- Pairwise judge with Elo-style aggregation ([GPT-4V as a text-to-3D
  evaluator](https://arxiv.org/abs/2401.04092)).
- **CLIP score** only as a secondary metric: it captures gist, not layout, proportions or block quality,
  and blocky renders are off its training distribution. Keep it only if it tracks human ratings. Never the
  in-loop critic (no actionable feedback). Image–image similarity (CLIP or DINOv2) between a reference and
  a `ref_view` render can feed `reference_match`.
- MCP server for the tool layer (official Python SDK), giving a baseline: a generic client (Claude Desktop
  or Claude Code) driving the same tools vs. this harness. Optionally run VibeCraft on a few prompts.
- Ablations: low-level vs. semantic tools; model A vs. B; with vs. without recipes; annotated renders;
  code-as-action.

## 17. Related work

What each contributes; **B** = borrowed, **A** = avoided.

**Minecraft agents and building**
- **Voyager** ([arXiv](https://arxiv.org/abs/2305.16291)): Mineflayer bot; LLM writes JavaScript; skill
  library; self-verification; learns from execution errors. B: skill library, error feedback. A: embodied
  block-by-block action for terrain.
- **Mindcraft**: Mineflayer bot with fixed commands plus sandboxed code. B: tools + code escape hatch.
- **MineDojo**: benchmark/simulator with a learned reward model (MineCLIP). B: evaluation as a first-class part.
- **yuniko-software/minecraft-mcp-server**: Mineflayer MCP server, block-level tools; the loop lives in the
  client. B: MCP packaging. A: block-level tools, no own runtime.
- **VibeCraft** (amenti-labs): MCP server + Fabric client mod; Claude Code with skills as the harness;
  preset terrain generators (`generate_terrain(type="rolling_hills")`), WorldEdit passthrough, a
  `build(code)` sandbox. B: command sanitizer, curated material knowledge. A: preset generators too coarse
  for specific landscapes; no own runtime or evaluation.
- **GDMC** and **GDPC** ([docs](https://gdpc.readthedocs.io/en/stable/)): competition for procedural
  settlements adapted to terrain, judged on adaptability, functionality, narrative, aesthetics; GDPC edits
  live worlds through the GDMC HTTP Interface mod. B: judging criteria; possible live backend.
- **MineAnyBuild** ([arXiv](https://arxiv.org/abs/2505.20148)): spatial-planning benchmark for Minecraft
  building. B: eval design; supports keeping geometry in code.

**LLM agents building 3D scenes**
- **SceneCraft** ([arXiv](https://arxiv.org/abs/2403.01248), ICML 2024): LLM writes Blender code; plans a
  relational scene graph; relations become scoring functions optimized for layout (a matrix of position,
  scale, orientation per asset); a GPT-4V reviewer revises from renders; an outer loop learns a
  non-parametric function library; CLIP for asset retrieval and as a metric; constraint passing score.
  B: feature graph, constraint passing, library learning. A: code as the main action; LLM-rewritten
  scoring functions; CLIP as the headline metric.
- **3D-GPT** ([arXiv](https://arxiv.org/abs/2310.12945)): agents expand a brief prompt, then choose
  parameters for procedural generators (Infinigen). B: the core idea (LLM picks procedural parameters);
  prompt expansion. Infinigen's terrain code is an algorithm reference.
- **Holodeck** ([arXiv](https://arxiv.org/abs/2312.09067), CVPR 2024): LLM outputs spatial relations; a
  constraint solver places objects. B: relational placement for scattering and photo-derived layouts.
- **BlenderAlchemy** ([arXiv](https://arxiv.org/abs/2404.17672), ECCV 2024): candidate edits rendered and
  compared by a VLM; revert when nothing improves. B: comparison-based acceptance; later, sampling 2–3
  candidate variants per revision.
- **CADCodeVerify** ([arXiv](https://arxiv.org/abs/2410.05340), ICLR 2025): verification questions from
  the request, answered on renders. B: spec checklist.

**Evaluation and grounding**
- **TIFA** ([arXiv](https://arxiv.org/abs/2303.11897)), **Davidsonian Scene Graph**
  ([arXiv](https://arxiv.org/abs/2310.18235)): prompt → atomic questions (DSG: with dependencies). B:
  checklist structure and dependency skipping.
- **Set-of-Mark** ([arXiv](https://arxiv.org/abs/2310.11441)): marks on regions improve VLM grounding. B:
  annotated renders.
- **GPT-4V as a text-to-3D evaluator** ([arXiv](https://arxiv.org/abs/2401.04092)): pairwise comparisons +
  Elo, checked against humans. B: judge design and calibration.
- **Self-Refine** ([arXiv](https://arxiv.org/abs/2303.17651)), **Reflexion**
  ([arXiv](https://arxiv.org/abs/2303.11366)), **CRITIC** ([arXiv](https://arxiv.org/abs/2305.11738)),
  **Huang et al.** ([arXiv](https://arxiv.org/abs/2310.01798)): refinement works best grounded in tools.
  B: checks before the VLM; fresh-context critic.
- **Xiong et al.** ([arXiv](https://arxiv.org/abs/2306.13063)): verbalized confidence is poorly
  calibrated. B: objective stage gates.

**Agent engineering**
- **SWE-agent** ([arXiv](https://arxiv.org/abs/2405.15793)): the agent–computer interface matters; reject
  invalid edits with explanations. B: tool principles and guardrails.
- **CodeAct** ([arXiv](https://arxiv.org/abs/2402.01030)): code as the action space. B: escape hatch.
- Anthropic: [Building effective agents](https://www.anthropic.com/research/building-effective-agents),
  [Writing effective tools for agents](https://www.anthropic.com/engineering/writing-tools-for-agents),
  [Effective context engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents);
  Thorsten Ball's [How to Build an Agent](https://ampcode.com/how-to-build-an-agent) (the loop's starting point).

**Terrain generation**
- **Hnaidi et al. 2010** ([paper](https://diglib.eg.org/handle/10.2312/CGF.v29i7pp2179-2186)): terrain
  from feature curves (ridges, rivers, cliffs) with attributes. B: features as first-class objects.
- **Guérin et al. 2017** ([paper](https://www.cs.purdue.edu/homes/bbenes/papers/Guerin17ToG.pdf)):
  terrain from sketched ridges and rivers. B: sketch input.

**Positioning:** existing Minecraft agents act block by block or pick preset generators, with the loop in
a generic client. This project treats terrain as a canvas of semantic, replayable operations, with features
as the unit of planning and checking, and its own runtime, critic and evaluation.

## 18. Fallbacks

| Component | If it doesn't work well | Fallback |
|---|---|---|
| Blind comparison gate (§11.4) | still biased (> 90% revision wins, or position bias) | judge both orders; else gate on checks + checklist only and keep the latest revision within the round limit |
| Spec checklist (§11.3) | VLM items noisy | mark unverified instead of failed; rely on code-checked items |
| SPEC step (§9.3) | no benchmark gain in the SPEC on/off ablation | builder gets the raw prompt; keep the graph only as the critic/eval checklist |
| Feature attributes (§4) | ambiguous boundaries | dominant-feature map; skip that feature's checks, use global stats |
| Category thresholds (§4.2) | categories don't match how results look | recalibrate with the sweep; one category of slack |
| Annotated renders (§10.4) | labels clutter or don't help | outlines only, or per-feature close-ups |
| Reference images (§12) | SPEC misreads layout or the projection is wrong | rely on `reference_match` in the critique loop; later `ref_view`; ask for a sketch; if time is short, ship text-only |
| Photo slope matching | angles unreliable | categories and relative constraints only |
| Stage gates (§9.1) | stall | round limit; record why; continue with the best attempt |
| CLIP score (§16) | doesn't track human ratings | drop it |
| Recipe library (§14) | no gain or stale recipes | turn off; keep promoted presets |
| Live backend (§8) | unsupported on 26.3 | offline compile + in-game screenshots |
| L3 renders (§10.3) | no client available | L2 |
| Code-as-action (§15) | unsafe or unreliable | tools only; log the request as a missing-tool signal |
| PyVista (L2) | too slow or unavailable | moderngl/pyrender; numpy+PIL isometric renderer |

## 19. Repository layout

```
terraform-agent/
  CLAUDE.md
  pyproject.toml            # amulet-core==1.9.49; extras: dev, render
  docs/                     # DESIGN.md, ROADMAP.md, EVIDENCE.md, images/ (README shots)
  terrain/
    config.py               # WORLD_MIN_Y, WORLD_MAX_Y
    state.py                # WorldState, features
    features.py             # measured feature attributes, categories (§4.2)
    opstack.py              # op stack, replay, checkpoints
    brushes.py              # distances, falloff, blend
    ops/                    # registry.py, region.py, character.py, terrain/water/materials ops
    materials.py            # palette rule engine, block list + looks
    compiler/               # canvas → volume → patch
    backends/               # base.py, amulet_backend.py, memory_backend.py
    render/                 # l1.py, l2.py, block_colors.json
  agent/
    models.py               # adapter(s), prices
    tools.py                # op tools + observe/control tools
    session.py              # agent-facing wrapper over the op stack
    harness.py              # outer + inner loop, acceptance gate
    critic.py               # checks, critique, blind comparison
    context.py              # episodes, system prompt, scratchpad
    budget.py, tracing.py, run.py
  eval/                     # benchmark.yaml, run_eval.py, judge.py, reports/
  scripts/                  # hello_amulet, demos, compile_world, calibration_sweep, migrate_stacks, build_block_colors,
                            # gate_stats, gate_check
  tests/                    # ops/, agent/ (fake model adapter), backends, compiler, render
  traces/ renders/ worlds/  # gitignored run outputs
```

## 20. Decisions and open questions

**Decided**
- Minecraft Java 26.3; amulet-core 1.9.49.
- Heightmap canvas + op stack; local voxel pass for 3D features later.
- Offline Amulet backend for development and evals; live backend for demos only.
- Structured palette rules, not expressions.
- Episodes rebuilt from state, not history edits.
- L2 via PyVista; L3 via in-game screenshots with L2 as the fallback.
- Features as the unit of spec, checks and critique; categories owned by code.
- Text prompts and reference images share one SPEC step and feature graph; the eval checks builds against
  hand-written expected graphs.
- Acceptance: objective checks first, then a blind comparison; never gate on absolute rubric scores or
  model confidence.

**Open**
- Calibration values for the 0–1 scales and the category ranges.
- Whether GDPC supports 26.3 (live backend option).
- Second provider for the critic and judge.
- UI for sketch input and mid-run feedback.
