# =============================================================
# PACKAGE: scanner  (top‑level, alongside prediction_engine/)
# =============================================================
# This commit bootstraps the **scanner** package described in the
# blueprint.  It is self‑contained, unit‑testable, and imports only
# pandas / numpy / asyncio / pyarrow (already present in the repo).
# ────────────────────────────────────────────────────────────────
# Directory layout
#   scanner/
#       __init__.py
#       rules.py           # pure boolean masks
#       detectors.py       # Detector objects wrapping rules
#       recorder.py        # DataGroupBuilder – persists snapshots
#       live_loop.py       # asyncio live runner
#       backtest_loop.py   # historical iterator
#   scripts2/run_scanner.py # CLI entry‑point (live / backtest)
# =============================================================

# ---------------------------------------------------------------------------
# FILE: scanner/__init__.py
# ---------------------------------------------------------------------------
"""Stock‑scanner package – emits *DataGroup* events consumed by
prediction_engine.* modules.  Import hierarchy is *flat* so
`from scanner import ScannerLoop` works across projects.
"""
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("scanner")
except PackageNotFoundError:  # locally editable checkout
    __version__ = "0.1.0-dev"

from .live_loop import ScannerLoop  # noqa: F401 re‑export for convenience
from .backtest_loop import BacktestScannerLoop  # noqa: F401
from .detectors import (  # noqa: F401
    GapDetector,
    HighRVOLDetector,
    #BullishMomentumDetector,
    CompositeDetector,
)
from .recorder import DataGroupBuilder  # noqa: F401

__all__ = [
    "ScannerLoop",
    "BacktestScannerLoop",
    "GapDetector",
    "HighRVOLDetector",
    #"BullishMomentumDetector",
    "CompositeDetector",
    "DataGroupBuilder",
]
