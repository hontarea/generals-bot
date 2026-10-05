"""Orchestrator: N matches in parallel between two bots -> results dir + report.

    python -m evaluation.evaluate <botA_dir> <botB_dir> \
        (--suite smoke | --seeds "0-19,42" | --seeds "random:30") \
        [--workers 8] [--no-replays] [--tag "castle-spacing-fix"] \
        [--enforce-timing] [--budget-ms 150] [--hard-cap-s 5] [--out DIR]

Every seed is played twice (seats swapped) — spawns are not symmetric, so a
seed can structurally favor one seat; both-color pairing cancels that.

Results dir (append-only store):
    evaluation/results/<date>_<botA>_vs_<botB>_<suite|tag>/
        config.json     exact invocation: bots, seeds, git hash, options
        games.jsonl     one result record per game, written as games finish
        report.txt      human-readable summary (regenerated even on Ctrl-C)
        replays/        one .npz per game (unless --no-replays)

Interrupted or crashed runs lose nothing: re-running with --out <dir> skips
(seed, swap) pairs already present in games.jsonl.

Score convention: win=1, draw=0.5, loss=0 for bot A. Decision rule: accept a
candidate when its score against the incumbent is above 0.5 AND the 95%
Wilson interval excludes 0.5. The smoke suite's 40 games give a wide interval
— enough to kill a clearly worse change, not enough to bless a marginal one.
A smoke score that does not clear the rule is undecided, not a rejection:
re-run with more seeds (--seeds "0-99") before concluding anything.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import evaluation  # noqa: F401  (sys.path bootstrap)
from evaluation.run_match import MatchJob, build_bot, run_match
from evaluation.seeds import resolve_seeds

RESULTS_ROOT = evaluation.REPO_ROOT / "evaluation" / "results"


@dataclass
class EvalOpts:
    workers: int
    replays: bool = True
    enforce_timing: bool = False
    budget_ms: float = 150.0
    hard_cap_s: float = 5.0
    rlimit_seat_a: int | None = None   # RLIMIT_AS for bot A's seat, in KB
    quiet: bool = False


# ---------------------------------------------------------------- statistics

def wilson(p_hat: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion (draws enter the score as
    half-successes upstream). Hand-rolled — no scipy in this project."""
    if n == 0:
        return 0.0, 1.0
    denom = 1 + z * z / n
    center = (p_hat + z * z / (2 * n)) / denom
    half = z * math.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def score_bot_a(records: list[dict]) -> tuple[float, int, int, int]:
    """(score, wins, draws, losses) for bot A across records (seat-mapped)."""
    w = d = l = 0
    for r in records:
        seat_a = r["swap"]  # swap=1 -> bot A sits in seat 1
        if r["winner"] is None:
            d += 1
        elif r["winner"] == seat_a:
            w += 1
        else:
            l += 1
    n = w + d + l
    return ((w + 0.5 * d) / n if n else 0.0), w, d, l


# ---------------------------------------------------------------- worker side

def _worker_init() -> None:
    # Workers do the JAX work; keep them off the GPU so N of them don't fight
    # over one device, and pay import + JIT warmup once per worker.
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import matchup
    from generals import GeneralsEnv
    env = GeneralsEnv(mode="competition")
    matchup.make_board(env, 0)  # compiles generate_grid for one (h, w) combo


def _run_job(job: MatchJob) -> dict:
    return run_match(job)


# ------------------------------------------------------------- orchestration

def _git_hash() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"],
                             cwd=str(evaluation.REPO_ROOT),
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def _load_done(games_path: Path) -> tuple[list[dict], set[tuple[int, int]]]:
    records: list[dict] = []
    done: set[tuple[int, int]] = set()
    if games_path.exists():
        with games_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                records.append(rec)
                done.add((rec["seed"], rec["swap"]))
    return records, done


def run_evaluation(bot_a: Path, bot_b: Path, seeds: list[int], seed_meta: dict,
                   opts: EvalOpts, out_dir: Path,
                   executor: ProcessPoolExecutor | None = None) -> dict:
    """Importable core behind the CLI. Returns a summary dict.

    If `executor` is given it is shared (and NOT shut down here); otherwise a
    private pool is created for this run.
    """
    bot_a, bot_b = Path(bot_a).resolve(), Path(bot_b).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    replays_dir = out_dir / "replays"
    if opts.replays:
        replays_dir.mkdir(exist_ok=True)

    config = {
        "argv": sys.argv,
        "bot_a": str(bot_a), "bot_b": str(bot_b),
        "seeds": seed_meta,
        "git": _git_hash(),
        "started": datetime.now().isoformat(timespec="seconds"),
        "opts": {"workers": opts.workers, "replays": opts.replays,
                 "enforce_timing": opts.enforce_timing,
                 "budget_ms": opts.budget_ms, "hard_cap_s": opts.hard_cap_s,
                 "rlimit_seat_a": opts.rlimit_seat_a},
    }
    config_path = out_dir / "config.json"
    if config_path.exists():
        old = json.loads(config_path.read_text())
        if old["bot_a"] != config["bot_a"] or old["bot_b"] != config["bot_b"]:
            raise SystemExit(
                f"refusing to resume: {out_dir} was created for different "
                f"bots (see its config.json); pass a fresh --out")
        # The stored seed list is authoritative on resume — a re-run of
        # "random:N" would otherwise draw fresh seeds and break pairing.
        seed_meta = old["seeds"]
        seeds = list(seed_meta["seeds"])
    else:
        config_path.write_text(json.dumps(config, indent=2) + "\n")

    games_path = out_dir / "games.jsonl"
    records, done = _load_done(games_path)
    if done and not opts.quiet:
        print(f"[evaluate] resuming: {len(done)} finished games found in {games_path}")

    # Build each bot once, in the parent, before fanning out (concurrent
    # builds in the same dir from N workers would race).
    build_bot(bot_a)
    if bot_b != bot_a:
        build_bot(bot_b)

    jobs = []
    for seed in seeds:
        for swap in (0, 1):
            if (seed, swap) in done:
                continue
            rlimit = {}
            if opts.rlimit_seat_a is not None:
                rlimit = {swap: opts.rlimit_seat_a}  # bot A sits in seat `swap`
            jobs.append(MatchJob(
                seed=seed, swap=swap, bot_a_dir=bot_a, bot_b_dir=bot_b,
                replay_path=(replays_dir / f"s{seed:05d}_sw{swap}.npz"
                             if opts.replays else None),
                enforce_timing=opts.enforce_timing,
                budget_ms=opts.budget_ms, hard_cap_s=opts.hard_cap_s,
                rlimit_seat=rlimit))

    total = len(done) + len(jobs)
    label_a, label_b = bot_a.name, bot_b.name

    own_pool = executor is None
    if own_pool and jobs:
        import multiprocessing as mp
        executor = ProcessPoolExecutor(
            max_workers=opts.workers,
            mp_context=mp.get_context("spawn"),
            initializer=_worker_init)
    try:
        if jobs:
            pending = {executor.submit(_run_job, job): job for job in jobs}
            with games_path.open("a") as games_f:
                while pending:
                    finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for fut in finished:
                        job = pending.pop(fut)
                        exc = fut.exception()
                        if exc is not None:
                            # Harness bug, not a game result: quarantine it so a
                            # broken harness never masquerades as a bad bot.
                            with (out_dir / "harness_errors.log").open("a") as ef:
                                ef.write(f"seed={job.seed} swap={job.swap}: "
                                         f"{exc!r}\n")
                            if not opts.quiet:
                                print(f"[evaluate] HARNESS ERROR on seed "
                                      f"{job.seed} swap {job.swap}: {exc!r}",
                                      file=sys.stderr)
                            continue
                        rec = fut.result()
                        records.append(rec)
                        games_f.write(json.dumps(rec) + "\n")
                        games_f.flush()
                        if not opts.quiet:
                            _print_progress(records, total, label_a, rec)
    finally:
        if own_pool and executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        summary = write_report(records, out_dir, label_a, label_b, seed_meta)
    return summary


def _print_progress(records: list[dict], total: int, label_a: str, rec: dict) -> None:
    score, w, d, l = score_bot_a(records)
    lo, hi = wilson(score, len(records))
    seat_winner = ("draw" if rec["winner"] is None
                   else f"seat{rec['winner']} ({rec['bot0'] if rec['winner'] == 0 else rec['bot1']})")
    print(f"[{len(records):3d}/{total}] {label_a} {score:.3f} "
          f"CI {lo:.3f}-{hi:.3f}  W/D/L {w}/{d}/{l} | "
          f"s{rec['seed']:05d} sw{rec['swap']}: {rec['end_reason']} "
          f"{seat_winner} in {rec['turns']}t {rec['wall_s']:.1f}s")


# ------------------------------------------------------------------- report

def _bot_stat(records: list[dict], is_a: bool, key: str):
    """Per-game values of final.<key> for bot A (is_a) or bot B, seat-mapped."""
    out = []
    for r in records:
        seat = r["swap"] if is_a else 1 - r["swap"]
        out.append(r["final"][key][seat])
    return out


def _timing_line(records: list[dict], is_a: bool) -> str:
    p50s, p99s, maxs, faults = [], [], [], 0
    for r in records:
        seat = r["swap"] if is_a else 1 - r["swap"]
        pm = r["reply_ms"]
        if pm["p50"][seat] is not None:
            p50s.append(pm["p50"][seat])
            p99s.append(pm["p99"][seat])
            maxs.append(pm["max"][seat])
        faults += r["faults"][seat]
    if not p50s:
        return f"no timing data | faults {faults}"
    # median of per-game p50s, max of per-game p99s (conservative), global max;
    # exact global percentiles need a pass over the per-turn reply_ms in replays
    return (f"p50 {statistics.median(p50s):.1f}ms p99* {max(p99s):.1f}ms "
            f"max {max(maxs):.1f}ms | faults {faults}")


def write_report(records: list[dict], out_dir: Path, label_a: str,
                 label_b: str, seed_meta: dict) -> dict:
    n = len(records)
    score, w, d, l = score_bot_a(records)
    lo, hi = wilson(score, n)

    reasons = {k: 0 for k in ("capture", "deathtouch", "truncation", "crash",
                              "fault_forfeit")}
    for r in records:
        reasons[r["end_reason"]] += 1

    as_p0 = [r for r in records if r["swap"] == 0]
    as_p1 = [r for r in records if r["swap"] == 1]
    s_p0 = score_bot_a(as_p0)[0] if as_p0 else float("nan")
    s_p1 = score_bot_a(as_p1)[0] if as_p1 else float("nan")

    # Seeds bot A lost from BOTH seats: strong evidence of a genuine weakness
    # on that map type. (Seeds where the same seat won regardless of bot are
    # map-decided and carry little signal.)
    by_seed: dict[int, list[dict]] = {}
    for r in records:
        by_seed.setdefault(r["seed"], []).append(r)
    lost_both = sorted(
        seed for seed, rs in by_seed.items()
        if len(rs) == 2 and all(r["winner"] is not None and r["winner"] != r["swap"]
                                for r in rs))

    avg_len = statistics.mean(r["turns"] for r in records) if records else 0
    castles_a = statistics.mean(_bot_stat(records, True, "castles")) if records else 0
    castles_b = statistics.mean(_bot_stat(records, False, "castles")) if records else 0

    suite = seed_meta.get("suite") or seed_meta.get("expr", "?")
    lines = [
        f"{label_a}  vs  {label_b}        suite={suite}  {n} games  "
        f"{datetime.now():%Y-%m-%d %H:%M}",
        f"score({label_a}) = {score:.3f}  [{lo:.3f}, {hi:.3f}]   "
        f"W {w}  D {d}  L {l}",
        "decision rule: accept if score > 0.5 AND the 95% CI excludes 0.5. "
        "At 40 games the CI is wide — a smoke result that does not clear it "
        "is undecided, not a rejection; re-run with more seeds",
        "end reasons: " + " | ".join(f"{k} {v}" for k, v in reasons.items()),
        f"as p0: {s_p0:.3f}        as p1: {s_p1:.3f}        "
        f"(seat gap {abs(s_p0 - s_p1):.3f})",
        f"avg length {avg_len:.0f} turns   | {label_a} castles/game "
        f"{castles_a:.1f}  | {label_b} castles/game {castles_b:.1f}",
        f"timing {label_a}: {_timing_line(records, True)}",
        f"timing {label_b}: {_timing_line(records, False)}",
        f"seeds lost from both seats: "
        f"{', '.join(map(str, lost_both)) if lost_both else '(none)'}"
        + ("   <- look at these first" if lost_both else ""),
    ]
    report = "\n".join(lines) + "\n"
    (out_dir / "report.txt").write_text(report)

    return {"bot_a": label_a, "bot_b": label_b, "games": n, "score": score,
            "ci": [lo, hi], "wdl": [w, d, l], "end_reasons": reasons,
            "seat_scores": [s_p0, s_p1], "lost_both_seats": lost_both,
            "report": report, "out_dir": str(out_dir)}


# ---------------------------------------------------------------------- CLI

def default_workers() -> int:
    return max(1, (os.cpu_count() or 2) // 2)


def make_out_dir(bot_a: Path, bot_b: Path, suite_label: str,
                 tag: str | None) -> Path:
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    name = f"{stamp}_{bot_a.name}_vs_{bot_b.name}_{suite_label}"
    if tag:
        name += f"_{tag}"
    return RESULTS_ROOT / name


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bot_a", help="candidate bot directory (contains run.sh)")
    parser.add_argument("bot_b", help="opponent bot directory")
    parser.add_argument("--suite", default=None,
                        help="named seed suite from evaluation/seeds.py")
    parser.add_argument("--seeds", default=None,
                        help='seed expression: "0-19,42" or "random:50"')
    parser.add_argument("--workers", type=int, default=default_workers())
    parser.add_argument("--no-replays", action="store_true")
    parser.add_argument("--tag", default=None,
                        help="free-form label appended to the results dir name")
    parser.add_argument("--enforce-timing", action="store_true")
    parser.add_argument("--budget-ms", type=float, default=150.0)
    parser.add_argument("--hard-cap-s", type=float, default=5.0)
    parser.add_argument("--out", type=Path, default=None,
                        help="existing results dir to resume, or explicit target")
    args = parser.parse_args()

    bot_a, bot_b = Path(args.bot_a).resolve(), Path(args.bot_b).resolve()
    for d in (bot_a, bot_b):
        if not (d / "run.sh").exists():
            sys.exit(f"not a bot directory (no run.sh): {d}")

    try:
        seeds, seed_meta = resolve_seeds(args.suite, args.seeds)
    except ValueError as e:
        sys.exit(str(e))

    suite_label = args.suite or ("random" if seed_meta["random"] else "seeds")
    out_dir = args.out or make_out_dir(bot_a, bot_b, suite_label, args.tag)

    opts = EvalOpts(workers=args.workers, replays=not args.no_replays,
                    enforce_timing=args.enforce_timing,
                    budget_ms=args.budget_ms, hard_cap_s=args.hard_cap_s)
    try:
        summary = run_evaluation(bot_a, bot_b, seeds, seed_meta, opts, out_dir)
    except KeyboardInterrupt:
        print(f"\n[evaluate] interrupted — resume with:\n"
              f"  python -m evaluation.evaluate {args.bot_a} {args.bot_b} "
              f"{'--suite ' + args.suite if args.suite else '--seeds ' + repr(args.seeds)} "
              f"--out {out_dir}", file=sys.stderr)
        raise SystemExit(130)
    print()
    print(summary["report"], end="")
    print(f"results: {summary['out_dir']}")


if __name__ == "__main__":
    main()
