"""Evaluation harness: turns "run some games" into measurement.

Tools (run from the repo root):
    python -m evaluation.evaluate  — the smoke test: N matches in parallel,
                                     both seats per seed -> score + CI report
    python -m evaluation.run_match — one match -> result record (+ replay)

The harness wraps the existing competition machinery (competition/matchup.py,
competition/protocol.py) — it re-implements no game rules. matchup.py uses a
bare `from protocol import ...`, so importing it requires competition/ on
sys.path; the bootstrap below provides that for every harness module.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
_COMPETITION = REPO_ROOT / "competition"

for _p in (str(REPO_ROOT), str(_COMPETITION)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
