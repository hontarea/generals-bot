"""Worker: run exactly ONE match between two bot dirs -> one result record.

Wraps competition/matchup.py's machinery (make_board, make_transition) so the
harness plays by the same rules as the reference runner — no game logic here.
Additions over matchup: deterministic (seed, swap) setup, per-reply timing,
fault counting, crash-as-result semantics, and a compressed on-disk replay.

CLI:
    python -m evaluation.run_match <botA_dir> <botB_dir> --seed 17
        [--swap] [--replay out.npz] [--enforce-timing] [--budget-ms 150]
        [--hard-cap-s 5] [--no-build]
Prints the result record as JSON.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import evaluation  # noqa: F401  (sys.path bootstrap)
from evaluation.agent_io import AgentCrashed, AgentHung, AgentProc

END_REASONS = ("capture", "deathtouch", "truncation", "crash", "fault_forfeit")

REPLAY_FORMAT_VERSION = 1


@dataclass
class MatchJob:
    seed: int
    swap: int                                  # 1 -> bot B takes seat 0
    bot_a_dir: Path
    bot_b_dir: Path
    replay_path: Path | None = None            # None -> no replay written
    enforce_timing: bool = False
    budget_ms: float = 150.0
    hard_cap_s: float = 5.0
    rlimit_seat: dict[int, int] = field(default_factory=dict)  # seat -> RLIMIT_AS bytes


def _git_hash() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=str(evaluation.REPO_ROOT),
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def classify_end(is_done: bool, pre_step_time: int, winner_val: int,
                 deathtouch_turn: int | None) -> tuple[str, int | None]:
    """Map a finished (or truncated) game to (end_reason, winner seat).

    Any capture at/after the deathtouch threshold is definitionally a touch
    (deathtouch.py docstring), so the 800 boundary cleanly splits capture vs
    deathtouch. A mutual touch ends the game with winner -1 -> draw (null).
    """
    if not is_done:
        return "truncation", None
    winner = winner_val if winner_val >= 0 else None
    if deathtouch_turn is not None and pre_step_time >= deathtouch_turn:
        return "deathtouch", winner
    return "capture", winner


def _percentiles(samples: list[float]) -> dict:
    if not samples:
        return {"p50": None, "p99": None, "max": None}
    arr = np.asarray(samples)
    return {"p50": round(float(np.percentile(arr, 50)), 2),
            "p99": round(float(np.percentile(arr, 99)), 2),
            "max": round(float(arr.max()), 2)}


def run_match(job: MatchJob) -> dict:
    """Run one full match. Bot failures become results; only genuine harness
    bugs raise (the orchestrator quarantines those separately)."""
    # Heavy imports stay inside so the JAX-free orchestrator parent can
    # import this module for MatchJob without paying for the engine.
    import jax.numpy as jnp

    import matchup
    from generals import GeneralsEnv
    from generals.core import game

    t_start = time.perf_counter()
    env = GeneralsEnv(mode="competition")
    state = matchup.make_board(env, job.seed)
    H, W = (int(d) for d in state.armies.shape)
    transition = matchup.make_transition(env)
    get_obs = game.get_full_observation if env.perfect_info else game.get_observation

    bot_dirs = [Path(job.bot_a_dir), Path(job.bot_b_dir)]
    if job.swap:
        bot_dirs.reverse()
    labels = [d.name for d in bot_dirs]

    record: dict = {
        "seed": job.seed, "swap": int(bool(job.swap)),
        "bot0": labels[0], "bot1": labels[1],
        "h": H, "w": W,
    }

    # Per-turn logs
    reply_ms: list[list[float]] = [[], []]
    faults = [0, 0]
    recording = job.replay_path is not None
    if recording:
        snap_armies = [np.asarray(state.armies, dtype=np.int32)]
        snap_owner = [np.asarray(state.ownership, dtype=bool)]
        snap_castles = [np.asarray(state.castles, dtype=bool)]
        snap_winner = [int(state.winner)]
        log_actions_raw: list[np.ndarray] = []
        log_actions_eff: list[np.ndarray] = []
        log_faults: list[list[int]] = []
        log_ms: list[list[float]] = []

    seats: list[AgentProc] = []
    end_reason: str = "truncation"
    winner: int | None = None
    stderr_tail: dict[str, str] = {}
    info = None
    turn = 0
    try:
        for i, d in enumerate(bot_dirs):
            seats.append(AgentProc(d / "run.sh", i, H, W, label=labels[i],
                                   rlimit_as_bytes=job.rlimit_seat.get(i)))

        while turn < env.truncation:
            pre_step_time = int(state.time)
            replies = [None, None]
            failed_seat: int | None = None
            failure: str | None = None
            for i in (0, 1):
                obs = get_obs(state, i)
                try:
                    replies[i] = seats[i].ask(
                        obs, budget_ms=job.budget_ms,
                        hard_cap_s=job.hard_cap_s, enforce=job.enforce_timing)
                except AgentCrashed:
                    failed_seat, failure = i, "crash"
                    break
                except AgentHung:
                    failed_seat, failure = i, "fault_forfeit"
                    break
                reply_ms[i].append(replies[i].elapsed_ms)
                if replies[i].fault != 0:
                    faults[i] += 1
            if failed_seat is not None:
                end_reason = failure
                winner = 1 - failed_seat
                stderr_tail[str(failed_seat)] = seats[failed_seat].stderr_tail()
                break

            raw = np.stack([r.raw_action for r in replies])
            # what the engine is fed (post malformed/enforcement substitution);
            # builds are additionally rewritten to passes inside the transition
            eff = np.stack([r.action for r in replies])
            state, info = transition(state, jnp.asarray(eff))
            turn += 1

            if recording:
                snap_armies.append(np.asarray(state.armies, dtype=np.int32))
                snap_owner.append(np.asarray(state.ownership, dtype=bool))
                snap_castles.append(np.asarray(state.castles, dtype=bool))
                snap_winner.append(int(state.winner))
                log_actions_raw.append(raw)
                log_actions_eff.append(eff)
                log_faults.append([r.fault for r in replies])
                log_ms.append([r.elapsed_ms for r in replies])

            if bool(info.is_done):
                end_reason, winner = classify_end(
                    True, pre_step_time, int(info.winner), env.deathtouch_turn)
                break
    finally:
        for proc in seats:
            proc.close()

    if info is None:
        info = game.get_info(state)

    record.update({
        "winner": winner,
        "end_reason": end_reason,
        "turns": turn,
        "faults": faults,
        "reply_ms": {k: [_percentiles(reply_ms[0])[k], _percentiles(reply_ms[1])[k]]
                     for k in ("p50", "p99", "max")},
        "final": {
            "land": [int(x) for x in np.asarray(info.land)],
            "army": [int(x) for x in np.asarray(info.army)],
            "castles": [int((np.asarray(state.castles) & np.asarray(state.ownership[i])).sum())
                        for i in (0, 1)],
        },
        "wall_s": round(time.perf_counter() - t_start, 2),
    })
    if stderr_tail:
        record["stderr_tail"] = stderr_tail

    if recording:
        record["replay"] = Path(job.replay_path).name
        _write_replay(
            Path(job.replay_path), record, env, state,
            snap_armies, snap_owner, snap_castles, snap_winner,
            log_actions_raw, log_actions_eff, log_faults, log_ms)

    validate_record(record)
    return record


def _write_replay(path: Path, record: dict, env, final_state,
                  snap_armies, snap_owner, snap_castles, snap_winner,
                  log_actions_raw, log_actions_eff, log_faults, log_ms) -> None:
    meta = {
        "format_version": REPLAY_FORMAT_VERSION,
        "seed": record["seed"], "swap": record["swap"],
        "bot0": record["bot0"], "bot1": record["bot1"],
        "h": record["h"], "w": record["w"],
        "winner": record["winner"], "end_reason": record["end_reason"],
        "ruleset": {
            "mode": "competition",
            "truncation": int(env.truncation),
            "deathtouch_turn": env.deathtouch_turn,
            "build_castles": bool(env.build_castles),
            "perfect_info": bool(env.perfect_info),
        },
        "git": _git_hash(),
    }
    T = len(log_actions_raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        meta=np.array(json.dumps(meta)),
        mountains=np.asarray(final_state.mountains, dtype=bool),
        generals=np.asarray(final_state.generals, dtype=bool),
        general_positions=np.asarray(final_state.general_positions, dtype=np.int8),
        armies=np.stack(snap_armies),
        ownership=np.stack(snap_owner),
        castles=np.stack(snap_castles),
        winner=np.asarray(snap_winner, dtype=np.int8),
        actions_raw=(np.stack(log_actions_raw) if T else
                     np.zeros((0, 2, 5), dtype=np.int32)),
        actions_eff=(np.stack(log_actions_eff) if T else
                     np.zeros((0, 2, 5), dtype=np.int32)),
        fault_codes=(np.asarray(log_faults, dtype=np.uint8) if T else
                     np.zeros((0, 2), dtype=np.uint8)),
        reply_ms=(np.asarray(log_ms, dtype=np.float32) if T else
                  np.zeros((0, 2), dtype=np.float32)),
    )


# ------------------------------------------------------------ replay reader
# The reader lives beside the writer so the two can never drift. Replays are
# exact: re-simulating `actions_eff` from snapshot 0 through
# matchup.make_transition reproduces every later snapshot bit-for-bit, which
# `test_replay_roundtrip_resimulates` pins. That makes a replay a trustworthy
# record of what happened, not just a picture of it.


@dataclass
class Replay:
    meta: dict
    mountains: np.ndarray          # (h, w) bool
    generals: np.ndarray           # (h, w) bool
    general_positions: np.ndarray  # (2, 2)
    armies: np.ndarray             # (T+1, h, w) int32
    ownership: np.ndarray          # (T+1, 2, h, w) bool
    castles: np.ndarray            # (T+1, h, w) bool
    winner: np.ndarray             # (T+1,) int8
    actions_raw: np.ndarray        # (T, 2, 5) int32
    actions_eff: np.ndarray        # (T, 2, 5) int32
    fault_codes: np.ndarray        # (T, 2) uint8  0=ok 1=late 2=malformed
    reply_ms: np.ndarray           # (T, 2) float32

    @property
    def num_turns(self) -> int:
        return self.actions_raw.shape[0]

    @property
    def passable(self) -> np.ndarray:
        return ~self.mountains


def load_replay(path: Path | str) -> Replay:
    with np.load(path) as z:
        return Replay(
            meta=json.loads(str(z["meta"])),
            mountains=z["mountains"],
            generals=z["generals"],
            general_positions=z["general_positions"],
            armies=z["armies"],
            ownership=z["ownership"],
            castles=z["castles"],
            winner=z["winner"],
            actions_raw=z["actions_raw"],
            actions_eff=z["actions_eff"],
            fault_codes=z["fault_codes"],
            reply_ms=z["reply_ms"],
        )


def states_from_replay(rep: Replay):
    """Rebuild engine GameState objects, one per stored snapshot."""
    import jax.numpy as jnp

    from generals.core import game

    passable = jnp.asarray(rep.passable)
    mountains = jnp.asarray(rep.mountains)
    generals = jnp.asarray(rep.generals)
    general_positions = jnp.asarray(rep.general_positions.astype(np.int32))

    states, infos = [], []
    for t in range(rep.armies.shape[0]):
        ownership = jnp.asarray(rep.ownership[t])
        neutral = passable & ~ownership[0] & ~ownership[1]
        state = game.GameState(
            armies=jnp.asarray(rep.armies[t]),
            ownership=ownership,
            ownership_neutral=neutral,
            generals=generals,
            castles=jnp.asarray(rep.castles[t]),
            mountains=mountains,
            passable=passable,
            general_positions=general_positions,
            time=jnp.asarray(t, dtype=jnp.int32),
            winner=jnp.asarray(int(rep.winner[t]), dtype=jnp.int32),
            pool_idx=jnp.asarray(0, dtype=jnp.int32),
        )
        states.append(state)
        infos.append(game.get_info(state))
    return states, infos


def validate_record(rec: dict) -> None:
    """Schema check shared with tests — fail loudly on a malformed record."""
    required = {"seed", "swap", "bot0", "bot1", "winner", "end_reason", "turns",
                "faults", "reply_ms", "final", "wall_s", "h", "w"}
    missing = required - rec.keys()
    assert not missing, f"record missing keys: {missing}"
    assert rec["end_reason"] in END_REASONS, rec["end_reason"]
    assert rec["winner"] in (0, 1, None), rec["winner"]
    assert rec["swap"] in (0, 1)
    assert isinstance(rec["turns"], int) and rec["turns"] >= 0
    assert len(rec["faults"]) == 2
    assert set(rec["reply_ms"]) == {"p50", "p99", "max"}
    assert set(rec["final"]) == {"land", "army", "castles"}
    for k in ("land", "army", "castles"):
        assert len(rec["final"][k]) == 2
    if rec["end_reason"] in ("crash", "fault_forfeit"):
        assert rec["winner"] is not None
    json.dumps(rec)  # must be JSON-serializable


def build_bot(bot_dir: Path) -> None:
    """Run the bot's build.sh once if present (matchup.build_agent semantics,
    but raising instead of sys.exit so callers stay in control)."""
    build = Path(bot_dir) / "build.sh"
    if not build.exists():
        return
    result = subprocess.run(["bash", str(build)], cwd=str(build.parent))
    if result.returncode != 0:
        raise RuntimeError(f"build failed: {build}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bot_a", help="bot A directory (contains run.sh)")
    parser.add_argument("bot_b", help="bot B directory (contains run.sh)")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--swap", action="store_true",
                        help="bot B takes seat 0")
    parser.add_argument("--replay", type=Path, default=None,
                        help="write compressed replay to this .npz path")
    parser.add_argument("--enforce-timing", action="store_true",
                        help="replace over-budget replies with passes")
    parser.add_argument("--budget-ms", type=float, default=150.0)
    parser.add_argument("--hard-cap-s", type=float, default=5.0)
    parser.add_argument("--no-build", action="store_true",
                        help="skip running build.sh")
    args = parser.parse_args()

    bot_a, bot_b = Path(args.bot_a).resolve(), Path(args.bot_b).resolve()
    for d in (bot_a, bot_b):
        if not (d / "run.sh").exists():
            sys.exit(f"not a bot directory (no run.sh): {d}")
    if not args.no_build:
        build_bot(bot_a)
        if bot_b != bot_a:
            build_bot(bot_b)

    job = MatchJob(seed=args.seed, swap=int(args.swap),
                   bot_a_dir=bot_a, bot_b_dir=bot_b,
                   replay_path=args.replay,
                   enforce_timing=args.enforce_timing,
                   budget_ms=args.budget_ms, hard_cap_s=args.hard_cap_s)
    record = run_match(job)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
