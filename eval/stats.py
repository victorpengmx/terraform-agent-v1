"""Paired statistics for the ablations (DESIGN §16): exact sign test and a prompt-level bootstrap CI.

Runs pair up by (prompt, seed) across conditions. The sign test counts wins and losses (ties dropped). The
bootstrap resamples prompts, not runs: each prompt's difference is the mean over its seed pairs, so two seeds
of one prompt never count as independent evidence.
"""

from __future__ import annotations

import math
import random
from statistics import mean

BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 0


def sign_test(wins: int, losses: int) -> float:
    """Exact two-sided binomial p-value under p = 0.5 (ties excluded beforehand); 1.0 with no decisive pairs."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def bootstrap_ci(diffs_by_prompt: dict[str, list[float]], samples: int = BOOTSTRAP_SAMPLES, seed: int = BOOTSTRAP_SEED,
                 level: float = 0.95) -> tuple[float, float, float] | None:
    """(mean difference, low, high): prompts resampled with replacement, each contributing its mean paired
    difference. None without data."""
    per_prompt = [mean(v) for v in diffs_by_prompt.values() if v]
    if not per_prompt:
        return None
    rng = random.Random(seed)
    n = len(per_prompt)
    stats = sorted(mean(rng.choices(per_prompt, k=n)) for _ in range(samples))
    lo = stats[int((1 - level) / 2 * samples)]
    hi = stats[min(samples - 1, int((1 + level) / 2 * samples))]
    return mean(per_prompt), lo, hi


def label(ci: tuple[float, float, float] | None, a: str, b: str) -> str:
    """Plain-language verdict for "a minus b"."""
    if ci is None:
        return "no data"
    _, lo, hi = ci
    if lo <= 0 <= hi:
        return "no clear difference at this N"
    return f"{a} higher" if lo > 0 else f"{b} higher"


def percentile(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile (q in 0-100)."""
    if not values:
        return None
    s = sorted(values)
    return s[max(0, math.ceil(q / 100 * len(s)) - 1)]
