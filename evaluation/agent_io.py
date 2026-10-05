"""Subprocess agent I/O with timing, fault tracking, and hang protection.

Replaces matchup.py's blocking `ask_agent` for harness use:
  * every reply is timed (perf_counter around write->read);
  * malformed replies become passes and are counted as faults;
  * replies over the budget are counted as faults (and only replaced with
    passes when enforcement is explicitly on — matching real conditions,
    where the local runner measures but does not kill slow replies);
  * a reader thread + queue means a hung bot raises AgentHung after a hard
    cap instead of deadlocking the harness forever;
  * stderr is drained into a ring buffer so a crash record can carry the
    bot's last words.
"""
from __future__ import annotations

import os
import queue
import shlex
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import NamedTuple

import numpy as np

import evaluation  # noqa: F401  (sys.path bootstrap for `protocol`)
from protocol import encode_handshake, encode_observation

# Fault codes (also stored per turn in replays as uint8)
OK = 0
LATE = 1
MALFORMED = 2

PASS_ACTION = np.array([1, 0, 0, 0, 0], dtype=np.int32)

_STDERR_RING_LINES = 30


class Reply(NamedTuple):
    action: np.ndarray      # what the engine should be fed (post substitution)
    raw_action: np.ndarray  # as decoded from the wire (pass if undecodable)
    elapsed_ms: float
    fault: int              # OK | LATE | MALFORMED


class AgentCrashed(Exception):
    """Agent closed stdout / died mid-game."""


class AgentHung(Exception):
    """Agent produced no reply within the hard cap."""


class AgentProc:
    """One stdio bot subprocess speaking the competition protocol."""

    def __init__(self, run_sh: Path, player_id: int, H: int, W: int, *,
                 label: str, rlimit_as_bytes: int | None = None):
        self.label = label
        self.player_id = player_id
        run_sh = Path(run_sh)
        if rlimit_as_bytes is None:
            cmd = ["bash", str(run_sh)]
        else:
            # ulimit -v in the wrapper shell (KB) instead of preexec_fn, which
            # is unsafe in threaded parents.
            kb = max(1, rlimit_as_bytes // 1024)
            cmd = ["bash", "-c", f"ulimit -v {kb} && exec bash {shlex.quote(str(run_sh))}"]
        # Bots conventionally exec `python`; make sure the interpreter running
        # the harness resolves first, so an unactivated venv still works.
        child_env = os.environ.copy()
        child_env["PATH"] = f"{Path(sys.executable).parent}:{child_env.get('PATH', '')}"
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
            text=True,
            cwd=str(run_sh.parent),
            env=child_env,
        )
        self._stdout_q: queue.Queue[str | None] = queue.Queue()
        self._stderr_ring: deque[str] = deque(maxlen=_STDERR_RING_LINES)
        threading.Thread(target=self._drain_stdout, daemon=True).start()
        threading.Thread(target=self._drain_stderr, daemon=True).start()

        self.proc.stdin.write(encode_handshake(player_id, H, W))
        self.proc.stdin.flush()

    def _drain_stdout(self) -> None:
        for line in self.proc.stdout:
            self._stdout_q.put(line)
        self._stdout_q.put(None)  # EOF sentinel

    def _drain_stderr(self) -> None:
        for line in self.proc.stderr:
            self._stderr_ring.append(line.rstrip("\n"))

    @property
    def alive(self) -> bool:
        return self.proc.poll() is None

    def stderr_tail(self) -> str:
        return "\n".join(self._stderr_ring)

    def send_raw(self, text: str) -> None:
        """Write raw text to the agent's stdin (protocol fuzzing)."""
        self.proc.stdin.write(text)
        self.proc.stdin.flush()

    def read_line(self, timeout_s: float) -> str | None:
        """One stdout line within timeout, None on EOF; queue.Empty on timeout."""
        return self._stdout_q.get(timeout=timeout_s)

    def ask(self, obs, *, budget_ms: float = 150.0, hard_cap_s: float = 5.0,
            enforce: bool = False) -> Reply:
        """Send one observation frame, read one action line back.

        Raises AgentCrashed / AgentHung — the only two conditions the caller
        turns into a game result.
        """
        frame = encode_observation(obs)
        t0 = time.perf_counter()
        try:
            self.proc.stdin.write(frame)
            self.proc.stdin.flush()
        except (BrokenPipeError, ValueError, OSError) as e:
            raise AgentCrashed(f"{self.label}: stdin write failed: {e}") from None
        try:
            line = self._stdout_q.get(timeout=hard_cap_s)
        except queue.Empty:
            raise AgentHung(f"{self.label}: no reply within {hard_cap_s}s") from None
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if line is None:
            raise AgentCrashed(f"{self.label}: closed stdout mid-game "
                               f"(exit code {self.proc.poll()})")

        fault = OK
        raw = self._decode(line)
        if raw is None:
            fault = MALFORMED
            raw = PASS_ACTION.copy()
            action = raw
        else:
            action = raw
            if elapsed_ms > budget_ms:
                fault = LATE
                if enforce:
                    action = PASS_ACTION.copy()
        return Reply(action=action, raw_action=raw,
                     elapsed_ms=elapsed_ms, fault=fault)

    @staticmethod
    def _decode(line: str) -> np.ndarray | None:
        parts = line.split()
        if len(parts) != 5:
            return None
        try:
            return np.array([int(x) for x in parts], dtype=np.int32)
        except ValueError:
            return None

    def close(self) -> int | None:
        """EOF the agent (its exit signal), wait briefly, kill if needed."""
        try:
            self.proc.stdin.close()
        except (BrokenPipeError, ValueError, OSError):
            pass
        try:
            self.proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        return self.proc.poll()
