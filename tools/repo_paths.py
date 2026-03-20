from __future__ import annotations

from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parent.resolve()
DEFAULT_OPPONENT_REPO = (REPO_ROOT / "opponent" / "zhou").resolve()
