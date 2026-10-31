"""CLI entry point: wires the plumbing together and runs the harness on one prompt.

    python -m agent.run --prompt "a lone volcanic peak" [--image ref.jpg] [--model claude-haiku-5-5] [--seed 7]

--model and --critic-model take Anthropic (claude-...) or Gemini (gemini-..., needs the gemini extra) ids. SPEC
runs on the builder's model and the blind comparison on the critic's.

The trace goes to traces/<run_id>.jsonl, starting with a step-0 header record (prompt, reference image hashes,
models, budgets, canvas, git commit, tool schema hash) so a run can be reproduced or re-scored from its trace alone.
Every render shown to the builder
or critic is also saved under renders/<run_id>/. When the run ends (also on a budget stop or a crash), the
final op stack is saved to renders/<run_id>/stack.json with final_l1.png and final_l2_<view>.png next to it.
Compile it with `python -m scripts.compile_world --world <dir> --stack renders/<run_id>/stack.json`.

--render l2 (default): top-down relief map plus a voxel preview of the compiled blocks per observation (a few
seconds each, needs PyVista); the critic and the blind comparison need the 3D view to judge wall steepness,
cross-sections and materials. The preview is from the south-west corner, or with --image from due south
looking north, the viewpoint SPEC assumes for the photo (final renders then add final_l2_overview_s.png). l1: relief map only (fast). none: statistics only, and no render_view tool.
With l1 or l2 the builder can also call render_view mid-round (512 px, saved as renders/<run_id>/view_NNN.png).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

from agent.budget import Budget
from agent.critic import Critic
from agent.harness import RunResult, run_task
from agent.models import ModelAdapter, ModelConfig, make_adapter, price_known
from agent.session import Session
from agent.tools import Tool, control_tools, observe_tools, op_tools
from agent.tracing import Tracer, TraceRecord
from terrain.state import WorldState

EFFORTS = ["low", "medium", "high", "xhigh", "max"]
ENV_FILE = Path(__file__).resolve().parents[1] / ".env"  # repo root, regardless of the working directory


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the terraform agent on one prompt.")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--image", type=Path, action="append", default=[], help="reference image (repeatable)")
    ap.add_argument("--model", default=ModelConfig.model, help="builder (and SPEC) model id: claude-... or gemini-...")
    ap.add_argument("--effort", default=ModelConfig.effort, choices=EFFORTS)
    ap.add_argument("--critic-model", default=None, help="critic (and comparison) model id (default: same as --model)")
    ap.add_argument("--no-critic", action="store_true", help="single pass, no critique rounds")
    ap.add_argument("--rounds", type=int, default=3, help="max build/critique rounds")
    ap.add_argument("--size", type=int, nargs=2, default=(256, 256), metavar=("W", "D"), help="area in blocks")
    ap.add_argument("--base-y", type=float, default=64.0)
    ap.add_argument("--seed", type=int, default=None,
                    help="canvas seed (terrain noise, blind-comparison order); default: derived from the run id")
    ap.add_argument("--max-steps", type=int, default=Budget.max_steps)
    ap.add_argument("--max-cost", type=float, default=Budget.max_cost_usd, help="USD")
    ap.add_argument("--render", choices=["l1", "l2", "none"], default="l2", help="images shown to builder and critic")
    ap.add_argument("--no-spec", action="store_true",
                    help="skip the SPEC step: builder and critic get the raw prompt, no feature checklist (M6 ablation)")
    ap.add_argument("--compare-both-orders", action="store_true",
                    help="acceptance gate judges both orders and needs agreement (2x comparison cost; Opus and eval runs)")
    args = ap.parse_args()
    load_dotenv(ENV_FILE, override=False)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # model text (e.g. "≈") on a cp1252 Windows console

    cfg = RunConfig(prompt=args.prompt, images=tuple(args.image), model=args.model, effort=args.effort,
                    critic=not args.no_critic, critic_model=args.critic_model, rounds=args.rounds, size=tuple(args.size),
                    base_y=args.base_y, seed=args.seed, budget=Budget(max_steps=args.max_steps, max_cost_usd=args.max_cost),
                    render=args.render, use_spec=not args.no_spec, compare_both_orders=args.compare_both_orders)
    tracer = Tracer()
    print(execute(cfg, tracer, Path("renders") / tracer.run_id))


@dataclass(frozen=True)
class RunConfig:
    """One run's configuration: what `main` builds from the command line and the eval runner from a config."""

    prompt: str
    images: tuple[Path, ...] = ()  # reference image paths
    model: str = ModelConfig.model  # builder (and SPEC)
    effort: str | None = ModelConfig.effort
    critic: bool = True
    critic_model: str | None = None  # critic and comparison; None = the builder's model
    rounds: int = 3
    size: tuple[int, int] = (256, 256)
    base_y: float = 64.0
    seed: int | None = None  # None: derived from the run id (run_seed)
    budget: Budget = field(default_factory=Budget)
    render: str = "l2"  # l1 | l2 | none
    use_spec: bool = True
    compare_both_orders: bool = False


def execute(cfg: RunConfig, tracer: Tracer, out_dir: Path,
            make_model: Callable[[ModelConfig], ModelAdapter] = make_adapter) -> RunResult:
    """One run, in-process: session, tools, adapters, trace header and the harness; the final op stack and
    renders go to `out_dir` (also on a budget stop or a crash). `make_model` builds each role's adapter."""
    session = Session.flat(*cfg.size, base_y=cfg.base_y, seed=run_seed(cfg.seed, tracer.run_id))
    observe = [] if cfg.render == "none" else observe_tools(session, out_dir)
    tools = {t.name: t for t in [*op_tools(session), *control_tools(session), *observe]}
    model = make_model(ModelConfig(model=cfg.model, effort=cfg.effort))
    critic = Critic(make_model(ModelConfig(model=cfg.critic_model or cfg.model, effort=cfg.effort))) if cfg.critic else None
    for m in sorted({model.config.model, *([] if critic is None else [critic.model.config.model])}):
        if not price_known(m):
            print(f"WARNING: no price for {m} in agent/models.py PRICES: costs are reported as unknown and the "
                  f"cost budget can't be enforced (step and time budgets still apply)")
    images = [Path(p).read_bytes() for p in cfg.images]
    tracer.log(TraceRecord(run_id=tracer.run_id, stage="run", step=0, note="run header",
                           header=run_header(cfg, images, session, cfg.budget, tools)))

    print(f"run {tracer.run_id}: builder={cfg.model} critic={'off' if critic is None else critic.model.config.model} trace={tracer.path}")
    try:
        return run_task(
            prompt=cfg.prompt,
            images=images,
            session=session,
            tools=tools,
            model=model,
            tracer=tracer,
            budget=cfg.budget,
            critic=critic,
            render=None if cfg.render == "none" else make_renderer(cfg.render, out_dir, view=l2_view(images)),
            max_rounds=cfg.rounds,
            compare_both_orders=cfg.compare_both_orders,
            use_spec=cfg.use_spec,
        )
    finally:
        # The harness leaves the session on the latest accepted attempt, so this is the run's final terrain.
        views = ("overview_ne", "overview_sw", *(["overview_s"] if images else []))
        for path in save_final(session, out_dir, l2_views=views):
            print(f"saved {path}")


def run_seed(seed: int | None, run_id: str) -> int:
    """The canvas seed: `--seed` if given, else one derived from the run id, so every run differs and a
    seedless run can still be reproduced from its trace (the header records the seed either way)."""
    return seed if seed is not None else zlib.crc32(run_id.encode()) & 0x7FFFFFFF


def run_header(cfg: RunConfig, images: list[bytes], session: Session, budget: Budget, tools: dict[str, Tool]) -> dict[str, Any]:
    """What a run was asked and how it was configured: the trace's step-0 record.

    Reference images are recorded by path and sha256 (not embedded). `tools_sha256` hashes the tool list as
    the model sees it, so runs on different schemas can be told apart even from a dirty working tree.
    """
    base = session.stack.base
    tool_params = json.dumps([t.to_param() for t in tools.values()], sort_keys=True)
    return {
        "prompt": cfg.prompt,
        "references": [{"path": Path(p).as_posix(), "sha256": hashlib.sha256(b).hexdigest()} for p, b in zip(cfg.images, images)],
        "builder_model": cfg.model,
        "critic_model": (cfg.critic_model or cfg.model) if cfg.critic else None,
        "models": role_models(cfg),
        "effort": cfg.effort,
        "max_rounds": cfg.rounds,
        "compare_both_orders": cfg.compare_both_orders,
        "spec": "model" if cfg.use_spec else "none",
        "render": cfg.render,
        "budget": {"max_steps": budget.max_steps, "max_cost_usd": budget.max_cost_usd, "max_seconds": budget.max_seconds},
        "canvas": {"width": base.width, "depth": base.depth, "base_y": base.base_y, "seed": base.seed},
        "git": git_state(),
        "tools_sha256": hashlib.sha256(tool_params.encode()).hexdigest(),
    }


def role_models(cfg: RunConfig) -> dict[str, dict[str, str] | None]:
    """Provider and model per role. SPEC uses the builder's adapter, the blind comparison the critic's."""
    def entry(model: str) -> dict[str, str]:
        return {"provider": ModelConfig(model=model).provider, "model": model}

    critic = entry(cfg.critic_model or cfg.model) if cfg.critic else None
    return {"builder": entry(cfg.model), "spec": entry(cfg.model) if cfg.use_spec else None,
            "critic": critic, "comparison": critic}


def git_state() -> dict[str, Any] | None:
    """Commit and dirty flag of the repo this code runs from; None outside a git checkout."""
    root = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return {"commit": commit, "dirty": bool(dirty)}


def save_final(session: Session, out_dir: Path, l2_views: tuple[str, ...] = ("overview_ne", "overview_sw")) -> list[Path]:
    """Write the final op stack and its L1/L2 renders to `out_dir`; returns the paths written.

    The stack is written first so it survives a render failure. A failed L2 render (e.g. PyVista missing)
    is reported on stdout and skips only the L2 images.
    """
    from terrain.render.l1 import render_l1

    out_dir.mkdir(parents=True, exist_ok=True)
    written = [out_dir / "stack.json"]
    written[0].write_text(session.stack.to_json(), encoding="utf-8")
    title = f"final: {len(session.ops)} ops"
    (out_dir / "final_l1.png").write_bytes(render_l1(session.state, title=title))
    written.append(out_dir / "final_l1.png")
    try:
        from terrain.compiler import compile_state
        from terrain.render.l2 import render_volume

        volume = compile_state(session.state)
        for view in l2_views:
            path = out_dir / f"final_l2_{view}.png"
            path.write_bytes(render_volume(volume, view, title=f"{title} ({view})"))
            written.append(path)
    except Exception as e:  # noqa: BLE001 - the stack and L1 are already saved; say what is missing
        print(f"WARNING: final L2 render failed ({type(e).__name__}: {e}); saved the stack and L1 only")
    return written


def l2_view(references: list[bytes]) -> str:
    """The L2 camera for builder, critic and comparison renders. With reference images: due south looking
    north, the viewpoint SPEC assumes for a photo (agent/spec.py), so render and photo are comparable;
    otherwise the south-west corner, which shows two faces of every landform."""
    return "overview_s" if references else "overview_sw"


def make_renderer(level: str, out_dir: Path, view: str = "overview_sw") -> Callable[[WorldState, str | None], list[bytes]]:
    """Harness render callback: PNGs of a terrain, also saved to `out_dir` for inspection. A None title gives
    the renderers' neutral defaults (size and height range), used by the blind comparison. `view`: the L2
    camera preset (`l2_view`)."""
    from terrain.render.l1 import render_l1

    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    def render(state: WorldState, title: str | None) -> list[bytes]:
        nonlocal count
        images = [render_l1(state, title=title or "")]
        if level == "l2":
            from terrain.compiler import compile_state
            from terrain.render.l2 import render_volume

            images.append(render_volume(compile_state(state), view, title=title or view))
        for i, png in enumerate(images):
            (out_dir / f"{count:03d}_{i}.png").write_bytes(png)
        count += 1
        return images

    return render


if __name__ == "__main__":
    main()
