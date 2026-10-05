"""Named seed suites and seed-expression parsing.

A seed fully determines the generated map, so the same seeds give both bots
the same boards and map luck cancels — which is why `smoke` is frozen rather
than re-drawn per run. Whatever seeds a run uses, the exact resolved list is
recorded in that run's config.json, so any evaluation can be repeated or
compared like-for-like later.

If a smoke result looks too good, re-check it on a fresh `random:N` draw: a
seed set stared at daily invites tuning to its particular layouts.
"""
from __future__ import annotations

import os
import random

# Seeds drawn by "random:N" come from this range: above every named suite so
# fresh draws never collide with the frozen sets.
_RANDOM_SEED_LO = 10_000
_RANDOM_SEED_HI = 2**31

SUITES: dict[str, list[int]] = {
    # The smoke suite: 20 seeds x both seats = 40 games, ~1.5-3 min at 6
    # workers. The one frozen set, so scores are comparable across runs and
    # across bots. For anything longer, pass an explicit --seeds expression
    # ("0-99", "random:100"); the resolved list lands in the run's config.json
    # either way, so any evaluation stays repeatable like-for-like.
    "smoke": list(range(0, 20)),
}


def parse_seed_expr(expr: str, *, entropy: int | None = None) -> tuple[list[int], dict]:
    """Parse a seed expression into an explicit seed list.

    Forms:
        "17"              one seed
        "0-19"            inclusive range
        "0-19,42,100-110" ranges and singletons, mixed
        "random:50"       50 fresh distinct random seeds

    Returns (seeds, meta). meta records everything needed to reproduce the
    draw — evaluate.py dumps it verbatim into config.json, so "random:N"
    runs are always re-runnable from their logged seed list.
    """
    expr = expr.strip()
    if not expr:
        raise ValueError("empty seed expression")

    if expr.startswith("random:"):
        count_str = expr[len("random:"):]
        try:
            count = int(count_str)
        except ValueError:
            raise ValueError(f"bad random seed count: {count_str!r}") from None
        if count <= 0:
            raise ValueError(f"random seed count must be positive, got {count}")
        if entropy is None:
            entropy = int.from_bytes(os.urandom(8), "big")
        rng = random.Random(entropy)
        seeds = rng.sample(range(_RANDOM_SEED_LO, _RANDOM_SEED_HI), count)
        return seeds, {"expr": expr, "random": True, "entropy": entropy, "seeds": seeds}

    seeds: list[int] = []
    seen: set[int] = set()
    for part in expr.split(","):
        part = part.strip()
        if not part:
            raise ValueError(f"empty item in seed expression {expr!r}")
        lo, sep, hi = part.partition("-")
        try:
            if sep:
                lo_i, hi_i = int(lo), int(hi)
            else:
                lo_i = hi_i = int(lo)
        except ValueError:
            raise ValueError(f"bad seed item {part!r} in {expr!r}") from None
        if lo_i < 0:
            raise ValueError(f"negative seed in {part!r}")
        if hi_i < lo_i:
            raise ValueError(f"reversed range {part!r}")
        for s in range(lo_i, hi_i + 1):
            if s not in seen:
                seen.add(s)
                seeds.append(s)
    return seeds, {"expr": expr, "random": False, "entropy": None, "seeds": seeds}


def resolve_seeds(suite: str | None, seeds_expr: str | None,
                  *, entropy: int | None = None) -> tuple[list[int], dict]:
    """Resolve exactly one of (--suite, --seeds) into a seed list + meta."""
    if (suite is None) == (seeds_expr is None):
        raise ValueError("give exactly one of --suite or --seeds")
    if suite is not None:
        if suite not in SUITES:
            raise ValueError(f"unknown suite {suite!r}; known: {sorted(SUITES)}")
        seeds = list(SUITES[suite])
        return seeds, {"suite": suite, "random": False, "entropy": None, "seeds": seeds}
    seeds, meta = parse_seed_expr(seeds_expr, entropy=entropy)
    meta["suite"] = None
    return seeds, meta
