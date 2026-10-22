"""Rebuild the main9 analysis figures from the saved evaluation CSV files.

This entry point lives beside the generated figures.  The plotting
implementation is kept with the other five-bar analysis programs two
directories above.
"""

from __future__ import annotations

import sys
from pathlib import Path


FIVE_BAR_DIR = Path(__file__).resolve().parents[2]
if str(FIVE_BAR_DIR) not in sys.path:
    sys.path.insert(0, str(FIVE_BAR_DIR))

from plot_main9_model_analysis import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
