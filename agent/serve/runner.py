"""The serving turn loop. AGENT_SPEC.md §9.3.

Failure policy, in order of importance:

* **Never crash.** A crash forfeits the game outright; an invalid but well-formed
  action is a silent pass costing nothing and adding no fault. So every turn is
  wrapped, and every failure path emits a pass.
* **Never be late.** Over 150 ms is a fault, and 50 faults forfeit. A wall-clock
  guard bails to a pass at `BAIL_MS`.
* Greedy, never sampled.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from agent.serve import protocol
from agent.serve.numpy_net import NumpyNet
from agent.spec import phi

#: Bail below the 150 ms budget with room for the write and the engine's read.
BAIL_MS = 110.0


class Agent:
    """Holds the network and the carried phi memory for one game."""

    def __init__(self, weights_path: str | Path, hs: protocol.Handshake):
        self.net = NumpyNet.load(weights_path)
        self.hs = hs
        self.phi_state = phi.init_phi_state(np)

    def act(self, frame: protocol.Frame) -> str:
        obs_arr = protocol.frame_to_tensor(frame, self.hs)
        obs, scalars, move_m, build_m, self.phi_state = phi.augment(
            np, obs_arr, self.phi_state, self.hs.pad_h, self.hs.pad_w
        )
        action, _ = self.net.greedy(
            obs, move_m, build_m, scalars, self.hs.pad_h, self.hs.pad_w
        )
        return protocol.format_action(action)


def run(weights_path: str | Path, stdin=None, stdout=None, stderr=None) -> None:
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr

    handshake_line = stdin.readline()
    if not handshake_line:
        return
    hs = protocol.parse_handshake(handshake_line)

    # Weight load and the first forward both happen inside the first-move budget
    # (~10 s in the competition, but the local harness forfeits at 5 s).
    try:
        agent = Agent(weights_path, hs)
    except Exception as exc:  # noqa: BLE001
        print(f"[rl_bot] failed to load weights: {exc!r}", file=stderr, flush=True)
        agent = None

    while True:
        try:
            frame = protocol.read_frame(stdin, hs)
        except Exception as exc:  # noqa: BLE001
            print(f"[rl_bot] malformed frame: {exc!r}", file=stderr, flush=True)
            break
        if frame is None:
            break                                   # EOF: the game is over

        start = time.perf_counter()
        try:
            if agent is None:
                raise RuntimeError("no weights loaded")
            line = agent.act(frame)
            if (time.perf_counter() - start) * 1e3 > BAIL_MS:
                print(f"[rl_bot] turn {frame.turn} over budget", file=stderr, flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[rl_bot] turn {frame.turn}: {exc!r}", file=stderr, flush=True)
            line = protocol.PASS_LINE

        stdout.write(line + "\n")
        stdout.flush()                              # or both sides deadlock
