"""
src/inference/predictor.py
==========================
EEGPredictor — loads the trained EEGModel checkpoint and runs
inference on a single sliding-window chunk.

Class alignment:
  The checkpoint was built with n_classes=4 (BCI-2a standard), but
  training data only contained Left (class 0) and Right (class 1).
  Classes 2 and 3 were never targeted → their logits are untrained.

  After the 4-class softmax, we extract only probs[0] and probs[1]
  and renormalise them to sum to 1.  All downstream logic (smoother,
  display, policy) works on this clean 2-class distribution.

  Checkpoint loading still uses n_classes=4 so the architecture
  exactly matches the saved weights.

Normalization:
  Per-window z-score per channel is applied at inference time.
  This is valid because the model learns relative patterns within
  a window and matches the normalization used during training.
"""

import numpy as np
import torch

from src.model.eeg_model import EEGModel
from src.config.settings import (
    N_CHANNELS,
    N_CLASSES_CHECKPOINT,
    ACTIVE_CLASSES,
    CLASS_TO_LABEL,
)


class EEGPredictor:
    """
    Parameters
    ----------
    model_path : str   — path to the .pth state-dict file
    n_channels : int   — must match the checkpoint (default 10)
    device     : str | None — 'cuda', 'cpu', or None (auto-detect)
    """

    def __init__(
        self,
        model_path: str,
        n_channels: int = N_CHANNELS,
        device:     str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load with n_classes=4 to exactly match the checkpoint architecture
        self.model = EEGModel(n_channels=n_channels, n_classes=N_CLASSES_CHECKPOINT)
        state = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

        self._active = ACTIVE_CLASSES   # [0, 1]

        print(
            f"[Predictor] Model loaded on {self.device}  "
            f"({n_channels} ch, {N_CLASSES_CHECKPOINT} raw classes → "
            f"renormalised to {len(self._active)} active classes: "
            f"{[CLASS_TO_LABEL[c] for c in self._active]})"
        )

    def predict(self, window: np.ndarray) -> dict:
        """
        Run one forward pass on a single EEG window.

        Parameters
        ----------
        window : np.ndarray  shape (n_channels, window_samples)
            Raw (preprocessed) EEG window from SlidingWindowExtractor.

        Returns
        -------
        dict with keys:
          probs_full  — np.ndarray (4,)   raw 4-class softmax (debug only)
          probs       — np.ndarray (2,)   renormalised [P(Left), P(Right)]
          left        — float  renormalised P(Left)
          right       — float  renormalised P(Right)
          class_id    — int    0 or 1   (argmax of renormalised probs)
          label       — str   "LEFT" or "RIGHT"
          confidence  — float max of renormalised probs

        Why renormalize?
          The 4-class softmax spreads mass across all 4 outputs.  Even if
          the model strongly prefers Left, probs[0] might only be ~0.5
          because the other three outputs each take a slice.  After
          extracting only probs[0] and probs[1] and renormalising, the
          two-class probabilities are comparable to a model that was
          trained end-to-end on 2 classes.
        """
        # ── 1. Per-channel z-score normalization ──────────────────────────────
        mean = window.mean(axis=1, keepdims=True)
        std  = window.std(axis=1,  keepdims=True) + 1e-8
        norm = (window - mean) / std                    # (n_channels, T)

        # ── 2. Shape: (n_channels, T) → (1, T, n_channels) ───────────────────
        x = torch.tensor(norm.T, dtype=torch.float32).unsqueeze(0).to(self.device)

        # ── 3. Forward pass — 4-class softmax ────────────────────────────────
        with torch.no_grad():
            logits     = self.model(x)                              # (1, 4)
            probs_full = torch.softmax(logits, dim=1).cpu().numpy()[0]  # (4,)

        # ── 4. Extract only the trained classes and renormalise ────────────────
        #       probs_full[2] and probs_full[3] are untrained → discard them
        raw_active = probs_full[self._active]           # [P_left_raw, P_right_raw]
        denom      = raw_active.sum() + 1e-8
        probs      = raw_active / denom                 # renormalised to sum=1.0

        class_id   = int(np.argmax(probs))              
        confidence = float(probs[class_id])

        return {
            "probs_full":  probs_full,                  # (4,) — debug
            "probs":       probs,                       # (2,) — [P(Left), P(Right)]
            "left":        float(probs[0]),
            "right":       float(probs[1]),
            "class_id":    class_id,
            "label":       CLASS_TO_LABEL.get(class_id, "UNKNOWN"),
            "confidence":  confidence,
        }
