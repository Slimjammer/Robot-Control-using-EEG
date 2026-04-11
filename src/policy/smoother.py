"""
src/policy/smoother.py
======================
DecisionSmoother — stabilises noisy per-window predictions.

Strategy (two-stage):
  1. Moving average of raw softmax probabilities over the last N windows.
  2. Confidence threshold — if the winning class probability is below
     the threshold *or* the winner is not Left/Right, return "STOP".

This mirrors what the BCI pipeline overview calls the
"Decision Smoothing Layer" — converting a noisy stream like
  [L, R, R, L, R]
into a stable, actionable decision.
"""

from collections import deque

import numpy as np

from src.config.settings import SMOOTHER_WINDOW, CONFIDENCE_THRESHOLD, N_ACTIVE_CLASSES


class DecisionSmoother:
    """
    Parameters
    ----------
    window    : int   — number of past windows to average (default 8)
    threshold : float — minimum confidence to issue a decision (default 0.62)
    """

    def __init__(self, window: int = SMOOTHER_WINDOW, threshold: float = CONFIDENCE_THRESHOLD):
        self.window    = window
        self.threshold = threshold
        # Each entry is a 2-element renormalised vector [P(Left), P(Right)]
        # (predictor already discards untrained classes 2 & 3)
        self._history: deque[np.ndarray] = deque(maxlen=window)

    def update(self, prediction: dict) -> str:
        """
        Add the latest prediction and return a smoothed decision.

        Parameters
        ----------
        prediction : dict — output of EEGPredictor.predict()

        Returns
        -------
        "LEFT" | "RIGHT" | "STOP"
        """
        self._history.append(prediction["probs"])

        # Not enough history yet → abstain
        if len(self._history) < self.window:
            return "STOP"

        # Average renormalised [P(Left), P(Right)] across last N windows.
        # probs are already 2-element (predictor stripped classes 2 & 3).
        avg_probs    = np.mean(self._history, axis=0)   # (2,)  always sums ≈1.0
        best_class   = int(np.argmax(avg_probs))        # 0=Left, 1=Right
        best_conf    = float(avg_probs[best_class])

        # Confidence gate — below threshold means the signal is ambiguous
        if best_conf < self.threshold:
            return "STOP"

        return "LEFT" if best_class == 0 else "RIGHT"

    @property
    def avg_probs(self) -> np.ndarray | None:
        """Current averaged probability vector, or None if history is empty."""
        if not self._history:
            return None
        return np.mean(self._history, axis=0)

    def reset(self):
        """Clear history (e.g. between subjects or sessions)."""
        self._history.clear()
