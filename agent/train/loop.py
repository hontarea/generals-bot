"""The training loop. AGENT_SPEC.md §8.

Per iteration: gate eval -> curriculum check -> pool refresh -> rollout -> GAE ->
diagnostics -> filter + schedules + update -> log, EMA, checkpoint -> free.

The gate runs *before* training on eval iterations so iteration 0 is a clean
baseline.
"""
from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom

from agent.train import curriculum as curr
from agent.train.checkpoint import SigtermCatcher, TrainState, load, save
from agent.train.config import PRESETS, Config, get_config
from agent.train.gae import prepare_advantages
from agent.train.loss import (
    flatten_batch,
    get_learning_rate,
    make_optimizer,
    make_update_fn,
    select_indices,
    set_learning_rate,
    uniform_magnet,
    update_ent_coef,
)
from agent.train.net import PolicyValueNet, param_count
from agent.train.rollout import collect_rollout, init_rollout_state, make_step_fn


def cosine(start: float, end: float, t: int, total: int) -> float:
    """Keyed on the **iteration counter**, computed on the host (§6).

    Not an optax schedule on optimizer-step count: that would depend on
    `adv_top_frac` through the number of minibatches, and confound every
    comparison across filter fractions.
    """
    frac = min(max(t / max(total, 1), 0.0), 1.0)
    return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * frac))


def ema_update(ema, params, decay: float):
    """Host-side, every iteration, no bias correction. Nothing in training reads
    it — **it is what gets submitted**."""
    return jax.tree.map(lambda e, p: decay * e + (1.0 - decay) * p, ema, params)


class CsvLogger:
    """CSV, not just stdout: the regime metrics only mean anything as a series.

    Two things this has to get right, both of which bite on resume:

    * When appending to an existing file, the **existing header wins**. Writing
      the new row's key order against an old header silently shifts every column.
    * Fields that appear only on some iterations (the gate block) must be in the
      header even when the first row written does not carry them.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = None
        self._fieldnames = None

        if self.path.exists() and self.path.stat().st_size:
            with self.path.open(newline="") as f:
                header = next(csv.reader(f), None)
            if header:
                self._fieldnames = header

    def log(self, row: dict) -> None:
        if self._file is None:
            if self._fieldnames is None:
                self._fieldnames = list(row)
            self._file = self.path.open("a", newline="")
            self._writer = csv.DictWriter(
                self._file, fieldnames=self._fieldnames, extrasaction="ignore"
            )
            if self.path.stat().st_size == 0:
                self._writer.writeheader()

        dropped = set(row) - set(self._fieldnames)
        if dropped:
            print(f"[loop] WARNING: metrics not in the CSV header, dropped: "
                  f"{sorted(dropped)}", flush=True)
        self._writer.writerow(row)
        self._file.flush()

    def close(self):
        if self._file:
            self._file.close()
            self._file = None


def behaviour_stats(rollout) -> dict:
    """§8 behaviour block: is the agent finding the economy at all?"""
    from agent.spec.constants import KIND_BUILD, KIND_PASS

    kinds = rollout.actions[..., 0]
    builds = kinds == KIND_BUILD
    n_games = rollout.actions.shape[1]

    turn = rollout.scalars[..., 0] * 1200.0
    first_build_turn = jnp.where(builds, turn, jnp.inf).min()

    return {
        "build_rate": float(jnp.mean(builds)),
        "builds_per_1k_steps": float(jnp.sum(builds) / max(n_games, 1) * 1000
                                     / max(rollout.actions.shape[0], 1)),
        "first_build_turn": float(first_build_turn) if jnp.isfinite(first_build_turn) else -1.0,
        "pass_rate": float(jnp.mean(kinds == KIND_PASS)),
        "post_deathtouch_frac": float(jnp.mean(rollout.scalars[..., 3])),
        "draw_rate": float(jnp.mean((rollout.winners < 0) & rollout.terminated)),
        "mean_turn": float(jnp.mean(turn)),
    }


def train(cfg: Config, out_dir: Path, resume: Path | None = None,
          max_iters: int | None = None, max_seconds: float | None = None) -> TrainState:
    """Train until `max_iters`, or until `max_seconds` of wall clock is nearly up.

    `max_seconds` exists because PBS cannot be relied on to give us time to
    checkpoint: the SIGTERM -> SIGKILL grace is the queue's `kill_delay`
    attribute, set by admins (commonly ~10 s) and not settable by a user. So the
    job stops *itself* with room to spare rather than hoping to catch a signal.
    `SigtermCatcher` stays as a backstop for the cases this misses.
    """
    t_wall_start = time.perf_counter()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = CsvLogger(out_dir / "metrics.csv")
    ckpt_path = out_dir / "checkpoint.eqx"
    done_marker = out_dir / "DONE"

    key = jrandom.PRNGKey(cfg.seed)
    key, net_key = jrandom.split(key)
    net = PolicyValueNet(cfg.model, key=net_key)
    params, static = eqx.partition(net, eqx.is_inexact_array)

    optimizer = make_optimizer(cfg)
    opt_state = optimizer.init(params)
    magnet_fn = uniform_magnet if not cfg.use_magnet else _load_magnet()
    update = make_update_fn(static, cfg, optimizer, magnet_fn)

    state = TrainState(
        params=params, ema_params=jax.tree.map(jnp.copy, params), opt_state=opt_state,
        iteration=0, stage_idx=0, iters_in_stage=0, last_gate_score=0.0,
        low_signal_iters=0, key=key, ent_coef=cfg.ent_coef_start,
    )
    if resume is not None and Path(resume).exists():
        state = load(Path(resume), state)
        print(f"[loop] resumed from {resume} at iteration {state.iteration}")

    total_iters = max_iters if max_iters is not None else cfg.num_iters

    # A chained job that lands after training already finished must not spend
    # ~50 s tracing 16 board sizes just to run an empty loop.
    if state.iteration >= total_iters:
        print(f"[loop] already at iteration {state.iteration}/{total_iters}; nothing to do")
        done_marker.write_text(f"{state.iteration}\n")
        return state

    print(f"[loop] {param_count(net):,} params  |  {cfg.rows} rows x {cfg.num_steps} steps "
          f"= {cfg.batch_size:,} transitions/iter  |  {cfg.n_keep // cfg.minibatch_size} "
          f"updates/iter")

    cs = curr.CurriculumState(
        state.stage_idx, state.iters_in_stage, state.last_gate_score, state.low_signal_iters
    )
    env = curr.make_stage_env(cs.stage_idx, cfg.pool_size)
    step_fn = make_step_fn(env, static, cfg.shaping_weights)
    key = state.key
    pool, states, phi_state, key = init_rollout_state(env, key, cfg.num_envs, cfg.pool_size)
    lo, hi = curr.assert_stage_distances(pool, cs.stage_idx)
    print(f"[loop] stage {cs.stage_idx}: generals distance in [{lo}, {hi}]")

    iter_seconds: list[float] = []
    stop_reason = "completed"

    with SigtermCatcher() as sigterm:
        for iteration in range(state.iteration, total_iters):
            t0 = time.perf_counter()
            # The gate block is written on every row, blank when not evaluated,
            # so it is present in the header whichever iteration logs first.
            row: dict = {
                "iteration": iteration, "stage": cs.stage_idx,
                "gate_score": "", "gate_mean_len": "", "gate_unfinished": "",
            }

            # --- gate, before training so iteration 0 is a clean baseline ----
            if iteration % cfg.gate_every == 0:
                key, gate_key = jrandom.split(key)
                model = eqx.combine(state.ema_params, static)
                gate = curr.evaluate_vs_expander(
                    env, model, pool, gate_key, cfg.gate_games
                )
                cs = cs._replace(last_gate_score=gate.score)
                row["gate_score"] = gate.score
                row["gate_mean_len"] = gate.mean_length
                row["gate_unfinished"] = gate.unfinished
                print(f"[loop] iter {iteration} gate vs Expander: {gate.score:.3f} "
                      f"(mean length {gate.mean_length:.0f}, "
                      f"{gate.unfinished:.0%} unfinished)")

            # --- curriculum --------------------------------------------------
            delta = 0
            if curr.should_advance(cs, cfg):
                delta = +1
            elif curr.should_step_back(cs, cfg):
                delta = -1
                print(f"[loop] stepping BACK from stage {cs.stage_idx}: no terminal "
                      f"signal for {cs.low_signal_iters} iterations")

            if delta:
                cs = curr.transition(cs, delta)
                # A fresh object, always: mutating the env is a silent no-op.
                env = curr.make_stage_env(cs.stage_idx, cfg.pool_size)
                step_fn = make_step_fn(env, static, cfg.shaping_weights)
                # In-flight games from the old stage must be discarded (§4.6).
                pool, states, phi_state, key = init_rollout_state(
                    env, key, cfg.num_envs, cfg.pool_size
                )
                lo, hi = curr.assert_stage_distances(pool, cs.stage_idx)
                assert int(jnp.max(states.time)) == 0, "in-flight games survived a transition"
                print(f"[loop] -> stage {cs.stage_idx}: distance in [{lo}, {hi}]")

            elif iteration % cfg.reset_pool_every == 0 and iteration > 0:
                key, pool_key = jrandom.split(key)
                pool, _ = env.reset(pool_key)

            # --- rollout -----------------------------------------------------
            rollout, states, phi_state, key = collect_rollout(
                env, step_fn, state.params, static, states, phi_state, key, pool,
                cfg.num_steps,
            )
            jax.block_until_ready(rollout.rews)
            t_rollout = time.perf_counter() - t0

            # --- advantages and diagnostics ----------------------------------
            adv, diag = prepare_advantages(
                rollout, cfg.gamma, cfg.gae_lambda, cfg.shaping_coef
            )
            cs = curr.tick(cs, diag.terminal_frac, cfg)

            # --- schedules and update ----------------------------------------
            lr = cosine(cfg.lr, cfg.final_lr, iteration, total_iters)
            # Either the open-loop cosine or the closed-loop controller, never a
            # blend: `ent_target = 0.0` keeps the original schedule exactly (D19).
            ent_coef = (
                state.ent_coef if cfg.ent_target > 0.0
                else cosine(cfg.ent_coef_start, cfg.ent_coef_end, iteration, total_iters)
            )
            opt_state = set_learning_rate(state.opt_state, lr)

            batch = flatten_batch(rollout, adv)
            sample_idx = select_indices(batch.advs, cfg.n_keep)
            key, upd_key = jrandom.split(key)
            params, opt_state, metrics = update(
                state.params, opt_state, batch, sample_idx, upd_key, ent_coef
            )

            ema_params = ema_update(state.ema_params, params, cfg.ema_decay)
            # The controller reads the entropy this update produced, so it acts on
            # a one-iteration delay. That is deliberate: entropy responds to
            # `ent_coef` over tens of iterations, and a faster loop would chase noise.
            next_ent_coef = (
                update_ent_coef(ent_coef, float(metrics["entropy"]), cfg)
                if cfg.ent_target > 0.0 else ent_coef
            )
            state = state._replace(
                params=params, ema_params=ema_params, opt_state=opt_state,
                iteration=iteration + 1, stage_idx=cs.stage_idx,
                iters_in_stage=cs.iters_in_stage, last_gate_score=cs.last_gate_score,
                low_signal_iters=cs.low_signal_iters, key=key,
                ent_coef=next_ent_coef,
            )

            # --- log ---------------------------------------------------------
            row.update({k: float(v) for k, v in metrics.items()})
            row.update(diag._asdict())
            row.update(behaviour_stats(rollout))
            row.update({
                "lr": get_learning_rate(opt_state), "ent_coef": ent_coef,
                "iter_seconds": time.perf_counter() - t0,
                "rollout_seconds": t_rollout,
            })
            logger.log(row)

            if iteration % 10 == 0:
                print(
                    f"[loop] {iteration:6d} | stage {cs.stage_idx} | "
                    f"ratio {row['first_ratio']:.4f} kl {row['approx_kl']:.4f} "
                    f"ent {row['entropy']:.3f} | term {diag.terminal_frac:.3f} "
                    f"mcEV {diag.mc_explained_variance:+.3f} | "
                    f"build {row['build_rate']:.4f} | {row['iter_seconds']:.1f}s",
                    flush=True,
                )

            if (iteration + 1) % cfg.ckpt_every == 0 or sigterm.requested:
                save(ckpt_path, state, cfg.model)
                print(f"[loop] checkpointed at iteration {state.iteration}")
            if sigterm.requested:
                stop_reason = "sigterm"
                break

            # Free the big arrays before the next iteration allocates its own.
            del rollout, batch, adv

            # --- wall-clock budget ------------------------------------------
            # Stop *before* starting an iteration we cannot finish. The first
            # iteration includes JIT compilation and is a bad predictor, so the
            # estimate uses the recent steady-state ones.
            iter_seconds.append(time.perf_counter() - t0)
            if max_seconds is not None:
                recent = iter_seconds[-5:] if len(iter_seconds) > 1 else iter_seconds
                estimate = max(recent)
                elapsed = time.perf_counter() - t_wall_start
                if elapsed + estimate >= max_seconds:
                    stop_reason = "time_budget"
                    print(f"[loop] wall-clock budget reached ({elapsed:.0f}s of "
                          f"{max_seconds:.0f}s, next iteration ~{estimate:.0f}s); "
                          f"stopping at iteration {state.iteration}", flush=True)
                    break

    save(ckpt_path, state, cfg.model)
    logger.close()

    # Archive a ready-to-submit copy of the EMA weights, tagged by iteration.
    # `checkpoint.eqx` is a rolling file that the next link overwrites, so
    # without this there would be no way to go back to "the bot as of link 2" —
    # and §7.4 makes choosing between checkpoints the whole submission decision.
    # 14 MB each, so a long chain costs a couple of hundred MB.
    try:
        from agent.serve.weights import save_weights

        out_path = out_dir / "weights" / f"iter_{state.iteration:07d}.safetensors"
        save_weights(eqx.combine(state.ema_params, static), cfg.model, out_path)
        print(f"[loop] archived EMA weights -> {out_path}")
    except Exception as exc:  # noqa: BLE001
        # Never let an archiving problem lose a finished link's training state;
        # the checkpoint above is already safely on disk.
        print(f"[loop] WARNING: could not archive weights: {exc!r}", flush=True)

    if state.iteration >= total_iters:
        done_marker.write_text(f"{state.iteration}\n")
        print(f"[loop] finished all {total_iters} iterations")
    else:
        print(f"[loop] stopped at iteration {state.iteration}/{total_iters} "
              f"({stop_reason}); resume with --resume {ckpt_path}")
    return state


def _load_magnet():
    from agent.train.magnet import expander_magnet

    return expander_magnet


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    # Choices come from the registry, so a new experiment preset needs no edit here.
    ap.add_argument("--preset", default="full", choices=sorted(PRESETS))
    ap.add_argument("--out", type=Path, default=Path("runs/rl_bot"))
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="stop cleanly and checkpoint once this much wall clock "
                         "has passed; set it below the job's walltime")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    jax.config.update("jax_default_matmul_precision", "tensorfloat32")

    cfg = get_config(args.preset)
    if args.seed is not None:
        cfg = cfg.replace(seed=args.seed)

    print(f"jax {jax.__version__}  devices: {jax.devices()}")
    state = train(cfg, args.out, resume=args.resume, max_iters=args.iters,
                  max_seconds=args.max_seconds)

    # Exit code tells the job script whether the chain should continue:
    # 0 = all iterations done, 64 = stopped early with a resumable checkpoint.
    total = args.iters if args.iters is not None else cfg.num_iters
    raise SystemExit(0 if state.iteration >= total else 64)


if __name__ == "__main__":
    # Run as `python -m agent.train.loop` from the repo root.
    main()
