# ---------------------------------------------------------------------------
# drift_monitor.py
# ---------------------------------------------------------------------------
"""Detect data drift via KL divergence & KS‑test."""
from __future__ import annotations

"""Model‑Drift Monitor with Retrain Trigger
==========================================
Tracks **prediction accuracy** and **KL divergence** over a rolling window of
recent trades.  Fires a *retrain signal* if either metric breaches configured
thresholds.

Key Features
------------
* Rolling window (default 500 trades) implemented via ``collections.deque``.
* Online updates: ``update(pred_prob, outcome)`` ingests one trade in O(1).
* KL divergence computed on 10‑bin histogram of predicted probabilities.
* Accuracy drop measured versus a long‑run baseline held in ``ref_acc``.
* Status enum: ``CLEAR`` → ``WARNING`` → ``RETRAIN_SIGNAL``.
* JSON checkpoint every 1 000 updates so state survives restarts.
* Helper ``should_skip_retrain(vol_spike)`` respects blueprint rule to pause
  during extreme volatility.
"""

import json, logging, threading, time          # ⇦ add threading & time
from collections import Counter, deque
from enum import Enum
from pathlib import Path
from typing import Deque, Tuple

import numpy as np
from scipy.stats import entropy

_log = logging.getLogger(__name__)

__all__ = ["DriftStatus", "DriftMonitor"]


class DriftStatus(str, Enum):
    CLEAR = "CLEAR"
    WARNING = "WARNING"
    RETRAIN_SIGNAL = "RETRAIN_SIGNAL"


class DriftMonitor:
    """Rolling drift monitor.

    Parameters
    ----------
    win_size : int
        Number of most‑recent trades to keep.
    acc_drop_th : float
        Relative accuracy drop (%) that triggers warning.
    kl_th : float
        KL‑divergence threshold.
    ckpt_path : Path | None
        If set, state is checkpointed every 1000 updates.
    """

    def __init__(
        self,
        win_size: int = 500,
        acc_drop_th: float = 0.10,
        kl_th: float = 0.15,
        ckpt_path: Path | None = Path("prediction_engine/artifacts/drift_state.json"),
    ) -> None:
        self.win_size = win_size
        self.acc_drop_th = acc_drop_th
        self.kl_th = kl_th
        self.hist: Deque[Tuple[float, bool]] = deque(maxlen=win_size)
        self.ref_acc: float | None = None  # long‑term baseline accuracy
        self.ckpt_path = ckpt_path
        self._updates = 0
        self._pending: dict[int, float] = {}
        self._counter: int = 0

        # load checkpoint if exists
        if ckpt_path and ckpt_path.exists():
            self._load()

    # ------------------------------------------------------------------
    '''def update(self, pred_prob: float, outcome: bool):
        """Add one trade outcome.

        Parameters
        ----------
        pred_prob : float
            Model‑predicted probability of *positive* outcome.
        outcome : bool
            True if trade made money, else False.
        """
        self.hist.append((float(pred_prob), bool(outcome)))
        self._updates += 1
        if self.ref_acc is None and len(self.hist) == self.win_size:
            self.ref_acc = self._accuracy()
        if self.ckpt_path and self._updates % 1000 == 0:
            self._save()'''

    # ------------------------------------------------------------------
    def update(self, pred_prob: float, outcome: bool):
        """
        Legacy one-shot helper that records a prediction **and**
        its outcome in one call.  Internally delegates to the
        two-stage API so everything goes through the same path.
        """
        tid = self.log_pred(pred_prob)        # ticket id → pending dict
        self.log_outcome(tid, outcome)        # attaches realised P/L


    # ------------------------------------------------------------------
    def status(self) -> Tuple[DriftStatus, dict]:
        if len(self.hist) < self.win_size // 2:
            return DriftStatus.CLEAR, {}

        cur_acc = self._accuracy()
        acc_drop = 0.0 if self.ref_acc is None else max(0.0, (self.ref_acc - cur_acc) / self.ref_acc)
        kl = self._kl_divergence()

        if acc_drop >= self.acc_drop_th or kl >= self.kl_th:
            return DriftStatus.RETRAIN_SIGNAL, {"acc_drop": acc_drop, "kl": kl}
        if acc_drop >= 0.5 * self.acc_drop_th or kl >= 0.5 * self.kl_th:
            return DriftStatus.WARNING, {"acc_drop": acc_drop, "kl": kl}
        return DriftStatus.CLEAR, {"acc_drop": acc_drop, "kl": kl}

    # ------------------------------------------------------------------
    def _accuracy(self) -> float:
        if not self.hist:
            return 0.0
        return sum(int(p > 0.5) == o for p, o in self.hist) / len(self.hist)

    def _kl_divergence(self) -> float:
        if not self.hist:
            return 0.0
        preds = np.fromiter((p for p, _ in self.hist), dtype=float)
        bins = np.linspace(0.0, 1.0, 11)
        p_hist, _ = np.histogram(preds, bins=bins, density=True)
        # reference: uniform target; could be long‑term hist instead
        q = np.full_like(p_hist, 1.0 / len(p_hist))
        return float(entropy(p_hist + 1e-9, q + 1e-9))


    # ------------------------------------------------------------------
    def _roll_and_checkpoint(self) -> None:
        """House-keeps counters & checkpoints every 1 000 updates."""
        self._updates += 1
        if self.ref_acc is None and len(self.hist) == self.win_size:
            self.ref_acc = self._accuracy()
        if self.ckpt_path and self._updates % 1000 == 0:
            self._save()


    # ------------------------------------------------------------------
    def _save(self):
        self.ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        with self.ckpt_path.open("w") as fp:
            json.dump({
                "hist": list(self.hist),
                "ref_acc": self.ref_acc,
                "updates": self._updates,
            }, fp)
        _log.info("DriftMonitor state checkpointed → %s", self.ckpt_path)

    def _load(self):
        with self.ckpt_path.open() as fp:
            data = json.load(fp)
        self.hist.extend(data.get("hist", []))
        self.ref_acc = data.get("ref_acc")
        self._updates = int(data.get("updates", 0))
        _log.info("DriftMonitor state loaded ← %s", self.ckpt_path)

    # ------------------------------------------------------------------
    def should_skip_retrain(self, vix_spike: bool) -> bool:
        """Return True if retrain should be skipped (volatility too high)."""
        return vix_spike

    # ------------------------------------------------------------------
    def reset(self):
        self.hist.clear()
        self.ref_acc = None
        self._updates = 0
        if self.ckpt_path and self.ckpt_path.exists():
            self.ckpt_path.unlink(missing_ok=True)

    def log_pred(self, prob: float) -> int:
        """Return ticket id for this prediction."""
        tid = self._counter
        self._pending[tid] = float(prob)
        self._counter += 1
        return tid

    def log_outcome(self, tid: int, made_money: bool):
        """Attach realised outcome to a prior prediction."""
        prob = self._pending.pop(tid, None)
        if prob is None:  # late or duplicate call: ignore
            return
        self.hist.append((prob, bool(made_money)))
        self._roll_and_checkpoint()

# ------------------------------------------------------------------
# Hook for external caller
# ------------------------------------------------------------------
GLOBAL_MONITOR: DriftMonitor | None = None

def get_monitor() -> DriftMonitor:
    """Return (or create) the process-wide DriftMonitor singleton."""
    global GLOBAL_MONITOR                # noqa: PLW0603
    if GLOBAL_MONITOR is None:
        GLOBAL_MONITOR = DriftMonitor()
    return GLOBAL_MONITOR



# ---------------------------------------------------------------------
if __name__ == "__main__":  # smoke test
    dm = DriftMonitor(win_size=200)
    rng = np.random.default_rng(42)
    # warm‑up
    for _ in range(200):
        dm.update(rng.random(), rng.random() > 0.5)
    print("Baseline", dm.status())
    # inject drift: model goes to 40‑% accuracy
    for _ in range(200):
        dm.update(rng.random(), rng.random() > 0.4)  # poorer accuracy
    print("After drift", dm.status())
