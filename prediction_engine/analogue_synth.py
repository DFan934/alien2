# ---------------------------------------------------------------------------
# prediction_engine/analogue_synth.py
# ---------------------------------------------------------------------------

from __future__ import annotations

"""Analogue Synthesiser
=======================

Given the *novel* feature vector **x₀**, a matrix of **nearest‑neighbour deltas**
```
ΔX = X_neigh − x₀  (shape = k × d)
```
AND the corresponding outcome differences ``Δy`` (shape = k,), compute a
non‑negative weight vector **β** such that

    ΔXᵀ · β  ≈  −δx₀   (inverse‑difference matching)
    Σ β = 1,   β ≥ 0.

The weighted average of neighbour outcomes (or their deltas) then serves as the
*synthetic exact match* outcome for **x₀**.

This implementation clamps pathological cases where the optimisation returns an
all‑zero weight vector, which in the previous version propagated NaNs downstream.

Place this file in  ``prediction_engine/analogue_synth.py``.
"""

from typing import Tuple

import numpy as np
from numpy.linalg import lstsq

try:
    from scipy.optimize import nnls  # type: ignore

    _HAS_SCIPY = True
except ModuleNotFoundError:  # pragma: no cover – CI without SciPy
    _HAS_SCIPY = False


class AnalogueSynth:
    """Static helpers for inverse‑difference analogue synthesis."""

    @staticmethod
    def weights(
            delta_mat: np.ndarray,
            target_delta: np.ndarray,
            *,
            var_nn: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute non‑negative weights that make ΔXᵀ·β ≈ target.

        Parameters
        ----------
        delta_mat
            Shape (k, d) matrix of neighbour‑minus‑novel deltas.
        target_delta
            Shape (d,) vector, typically ``-novel_feature`` so that the sum of
            weighted deltas cancels the novel deviation.

        Returns
        -------
        β : np.ndarray, shape (k,)
            Non‑negative weights summing to one.
            :param var_nn:
        """
        k, d = delta_mat.shape
        assert target_delta.shape == (d,), "target_delta dim mismatch"

        # Build augmented system enforcing Σβ = 1
        A_aug = np.vstack([delta_mat.T, np.ones(k)])  # (d+1) × k
        b_aug = np.append(target_delta, 1.0)  # (d+1,)

        if var_nn is not None:
            if var_nn.shape != (k,):
                raise ValueError("var_nn shape mismatch – expected (k,)")
            W = np.diag(1.0 / np.sqrt(var_nn + 1e-12))
            A_aug = W @ A_aug
            b_aug = W @ b_aug

        if _HAS_SCIPY:
            β, _ = nnls(A_aug, b_aug)
        else:
            β, *_ = lstsq(A_aug, b_aug, rcond=None)
            β = np.maximum(β, 0.0)

        s = β.sum()
        if s < 1e-8 or np.isnan(s):
            β = np.full(k, 1.0 / k, dtype=float)
        else:
            β /= s
        return β.astype(np.float32, copy=False)

    @staticmethod
    def synthesize(
        novel_outcome_proxy: float,
        neighbour_outcomes: np.ndarray,
        β: np.ndarray,
    ) -> float:
        """Return the *synthetic* outcome for the novel setup.

        This is simply the β‑weighted average of the neighbour outcomes, shifted
        by the novel proxy (e.g., current mark‑to‑market).  If your pipeline
        pre‑computes outcome deltas, pass zero for *novel_outcome_proxy*.
        """
        if neighbour_outcomes.shape != β.shape:
            raise ValueError("β/outcome dim mismatch")
        return novel_outcome_proxy + float(β @ neighbour_outcomes)


__all__: Tuple[str, ...] = ("AnalogueSynth",)
