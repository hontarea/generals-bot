#!/usr/bin/env python
"""Entry point for the RL submission.

Serves the trained policy as a pure-NumPy forward pass (no JAX at match time).
For a submission zip, copy `agent/spec` and `agent/serve` into `agent_pkg/agent/`
beside this file; from a checkout it imports `agent` from the repo root.
`run.sh` sets cwd to this directory.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# One dedicated core: extra BLAS threads only add contention and jitter.
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(var, "1")

# Prefer the bundled copy; fall back to the repo root when run from a checkout.
BUNDLED = HERE / "agent_pkg"
sys.path.insert(0, str(BUNDLED if (BUNDLED / "agent").is_dir() else HERE.parents[1]))

WEIGHTS = HERE / "weights.safetensors"


def main() -> None:
    from agent.serve.runner import run

    run(WEIGHTS)


if __name__ == "__main__":
    main()
