"""
src/inference/window.py
=======================
SlidingWindowExtractor — maintains an internal ring buffer.

Each call to `add(chunk)` appends new EEG samples.  Once the buffer
is full and enough new samples have arrived (step_samples), a copy of
the current window is returned — otherwise None is returned.

This drives the core of the real-time loop:
  chunk → add() → window or None → predictor.predict(window)
"""

import numpy as np


class SlidingWindowExtractor:
    """
    Parameters
    ----------
    window_samples : int  — total samples in one window (e.g. 1000 = 4 s)
    step_samples   : int  — samples between successive windows (e.g. 62 = 0.25 s)
    """

    def __init__(self, window_samples: int, step_samples: int):
        self.window_samples = window_samples
        self.step_samples   = step_samples

        self._buffer: np.ndarray | None = None   # (n_channels, window_samples)
        self._total_samples        = 0
        self._samples_since_yield  = step_samples  # trigger immediate yield when full

    def add(self, chunk: np.ndarray) -> np.ndarray | None:
        """
        Add a new chunk of shape (n_channels, n_new_samples).

        Returns
        -------
        window : np.ndarray  shape (n_channels, window_samples)  — or None
            A copy of the full buffer when a new step boundary is crossed.
            None while the buffer is still warming up or between steps.
        """
        n_channels, n_new = chunk.shape

        # Lazy initialise buffer on first call (avoids needing n_channels upfront)
        if self._buffer is None:
            self._buffer = np.zeros((n_channels, self.window_samples), dtype=np.float32)

        # Roll existing data left and insert new samples at the right end
        self._buffer = np.roll(self._buffer, -n_new, axis=1)
        self._buffer[:, -n_new:] = chunk

        self._total_samples       += n_new
        self._samples_since_yield += n_new

        # Yield once the buffer is fully populated AND a step boundary is crossed
        if (
            self._total_samples >= self.window_samples
            and self._samples_since_yield >= self.step_samples
        ):
            self._samples_since_yield = 0
            return self._buffer.copy()

        return None

    def reset(self):
        """Clear the buffer (e.g. between subjects)."""
        self._buffer              = None
        self._total_samples       = 0
        self._samples_since_yield = self.step_samples
