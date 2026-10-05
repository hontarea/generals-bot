"""Atomic checkpoint/resume. AGENT_SPEC.md §8.

MetaCentrum GPU queues cap walltime below the ~30 h a full run wants, so this has
to exist *before* the first long run, not after the first lost one.

Atomic means temp file + rename: a checkpoint half-written when the walltime
killer arrives must not overwrite the last good one.

Env states and phi are deliberately **not** persisted. Restarting the in-flight
games costs one stale rollout; serializing a pool of 20k boards would cost
gigabytes and a lot of fragile code.
"""
from __future__ import annotations

import dataclasses
import json
import os
import pickle
import signal
import tempfile
from pathlib import Path
from typing import NamedTuple

import jax
import equinox as eqx
import jax.numpy as jnp


class TrainState(NamedTuple):
    params: object
    ema_params: object
    opt_state: object
    iteration: int
    stage_idx: int
    iters_in_stage: int
    last_gate_score: float
    low_signal_iters: int
    key: jnp.ndarray
    #: Live value of the entropy controller (D19). It has to survive the handoff:
    #: a chain is eight jobs, and a controller that restarts from
    #: `ent_coef_start` on every link would sawtooth for the whole run. Defaulted,
    #: and read back with `.get`, so checkpoints written before D19 still load.
    ent_coef: float = 0.0


def save(path: Path, state: TrainState, model_cfg=None) -> None:
    """Write atomically: a killed process must not corrupt the last good file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    meta = {
        "iteration": state.iteration,
        "stage_idx": state.stage_idx,
        "iters_in_stage": state.iters_in_stage,
        "last_gate_score": state.last_gate_score,
        "low_signal_iters": state.low_signal_iters,
        "ent_coef": state.ent_coef,
    }
    # The model geometry travels with the weights. Without it, exporting a
    # checkpoint means guessing embed_dim/depth/n_head, and a wrong guess either
    # crashes or — worse — loads a differently-shaped net that still runs.
    if model_cfg is not None:
        meta["model"] = dataclasses.asdict(model_cfg)

    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    os.close(fd)
    try:
        with open(tmp, "wb") as f:
            f.write((json.dumps(meta) + "\n").encode())
            # opt_state holds optax NamedTuples that eqx cannot serialise on
            # their own, so the whole payload goes through eqx's leaf writer
            # with a pickled treedef.
            eqx.tree_serialise_leaves(
                f, (state.params, state.ema_params, state.opt_state, state.key)
            )
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def load(path: Path, like: TrainState) -> TrainState:
    """`like` supplies the pytree structure; only the leaves come off disk."""
    path = Path(path)
    with open(path, "rb") as f:
        meta = json.loads(f.readline().decode())
        params, ema_params, opt_state, key = eqx.tree_deserialise_leaves(
            f, (like.params, like.ema_params, like.opt_state, like.key)
        )

    return TrainState(
        params=params, ema_params=ema_params, opt_state=opt_state, key=key,
        iteration=meta["iteration"], stage_idx=meta["stage_idx"],
        iters_in_stage=meta["iters_in_stage"],
        last_gate_score=meta["last_gate_score"],
        low_signal_iters=meta["low_signal_iters"],
        ent_coef=meta.get("ent_coef", like.ent_coef),
    )


def read_meta(path: Path) -> dict:
    """The JSON header alone — iteration, curriculum counters, model geometry."""
    with open(path, "rb") as f:
        return json.loads(f.readline().decode())


def load_network(path: Path, which: str = "ema"):
    """Rebuild a `PolicyValueNet` from a training checkpoint.

    `which="ema"` is the default because the EMA is what gets submitted (§8);
    `which="raw"` gives the live training parameters.

    Structure is reconstructed from the checkpoint's own metadata, so exporting
    never depends on remembering which config produced a run.
    """
    import equinox as eqx
    import jax.random as jrandom

    from agent.train.config import ModelConfig, get_config
    from agent.train.loss import make_optimizer
    from agent.train.net import PolicyValueNet

    if which not in ("ema", "raw"):
        raise ValueError(f"which must be 'ema' or 'raw', got {which!r}")

    meta = read_meta(path)
    if "model" not in meta:
        raise ValueError(
            f"{path} has no model geometry in its header — it predates the "
            f"metadata change. Pass an explicit ModelConfig to rebuild it."
        )
    model_cfg = ModelConfig(**meta["model"])

    net = PolicyValueNet(model_cfg, key=jrandom.PRNGKey(0))
    params, static = eqx.partition(net, eqx.is_inexact_array)
    like = TrainState(
        params=params,
        ema_params=jax.tree.map(jnp.copy, params),
        opt_state=make_optimizer(get_config("full").replace(model=model_cfg)).init(params),
        iteration=0, stage_idx=0, iters_in_stage=0, last_gate_score=0.0,
        low_signal_iters=0, key=jrandom.PRNGKey(0),
    )

    state = load(path, like)
    chosen = state.ema_params if which == "ema" else state.params
    return eqx.combine(chosen, static), model_cfg, state


class SigtermCatcher:
    """Set a flag on SIGTERM so the loop can checkpoint at a clean boundary.

    Checkpointing from inside the handler would race the update that is mid-flight;
    the loop polls `.requested` between iterations instead.
    """

    def __init__(self):
        self.requested = False
        self._previous = {}

    def __enter__(self):
        for sig in (signal.SIGTERM, signal.SIGINT):
            self._previous[sig] = signal.signal(sig, self._handle)
        return self

    def __exit__(self, *_exc):
        for sig, handler in self._previous.items():
            signal.signal(sig, handler)
        return False

    def _handle(self, signum, _frame):
        self.requested = True
        print(f"\n[ckpt] signal {signum} received; checkpointing at the next "
              f"iteration boundary", flush=True)


def pickle_config(path: Path, cfg) -> None:
    """Store the config beside the checkpoint so a resume cannot silently use
    different hyperparameters."""
    Path(path).write_bytes(pickle.dumps(cfg))
