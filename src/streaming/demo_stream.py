"""
src/streaming/demo_stream.py
============================
DemoEEGStream — a clean, deterministic demo stream built from actual
pre-extracted MI epochs (not raw continuous EEG).

Why this is better for demonstration:
  - The raw continuous stream has long rest periods where ground truth
    is undefined (TrueLabel = --).
  - Demo stream stitches ONLY the labeled trial epochs together, so
    every window has a known, verifiable ground truth.
  - This uses the SAME ICA-cleaned, bandpass-filtered signal the model
    was trained on — so you get the best possible prediction quality.

Stream layout for mode='alternating':
  [rest_pad] [LEFT epoch] [rest_pad] [RIGHT epoch] [rest_pad] [LEFT ...] ...

All boundaries are tracked perfectly, so get_true_label() is exact.

Modes:
  'left'        -- only LEFT trials (good for testing LEFT bias)
  'right'       -- only RIGHT trials (good for testing RIGHT detection)
  'alternating' -- LEFT, RIGHT, LEFT, RIGHT ... (balanced view)
  'all'         -- all trials in their original dataset order
"""

import numpy as np
import mne
from mne.preprocessing import ICA

from src.config.settings import (
    SAMPLING_RATE,
    LOW_FREQ,
    HIGH_FREQ,
    STREAM_CHANNELS,
    BCI_CHANNEL_MAPPING,
    ANNOTATION_TO_CLASS,
    CLASS_TO_LABEL,
    WINDOW_SAMPLES,
    ACTIVE_CLASSES,
    bci2a_strip,
)


# Between-trial rest buffer — gives the sliding window time to clear
# after each trial before the next one starts
REST_SAMPLES = int(2.0 * SAMPLING_RATE)   # 2 seconds = 500 samples


class DemoEEGStream:
    """
    Parameters
    ----------
    gdf_path    : str   -- path to a training *T.gdf file
    mode        : str   -- 'left', 'right', 'alternating', 'all'
    n_trials    : int | None  -- limit to first N trials (None = all)
    rest_samples: int   -- silent buffer samples between trials
    apply_ica   : bool  -- apply ICA (should be True for best results)
    verbose     : bool  -- print trial inventory at startup
    """

    MODES = ("left", "right", "alternating", "all")

    def __init__(
        self,
        gdf_path:     str,
        mode:         str  = "alternating",
        n_trials:     int | None = None,
        rest_samples: int  = REST_SAMPLES,
        apply_ica:    bool = True,
        verbose:      bool = True,
    ):
        assert mode in self.MODES, f"mode must be one of {self.MODES}"

        self.mode         = mode
        self.rest_samples = rest_samples

        print(f"[DemoStream] Loading: {gdf_path}  (mode={mode})")
        epochs_left, epochs_right = self._load_epochs(gdf_path, apply_ica)

        if verbose:
            print(f"[DemoStream] Extracted  {len(epochs_left)} LEFT  and  {len(epochs_right)} RIGHT  epochs")

        # Build playlist of (epoch_data, label) tuples
        playlist = self._build_playlist(epochs_left, epochs_right, n_trials)
        print(f"[DemoStream] Playlist : {len(playlist)} trials  ->  {', '.join(t[1] for t in playlist)}")

        # Stitch into one continuous buffer with tracked label boundaries
        self.data, self.label_segments = self._stitch(playlist)
        self.n_channels = self.data.shape[0]
        self.n_samples  = self.data.shape[1]
        self.pointer    = 0

        total_sec = self.n_samples / SAMPLING_RATE
        print(f"[DemoStream] Ready   : {self.n_channels} ch  |  {self.n_samples} samples ({total_sec:.1f} s)")
        print(f"[DemoStream] Labeled : {len(self.label_segments)} segments  "
              f"({sum(1 for s in self.label_segments if s['label']=='LEFT')} LEFT, "
              f"{sum(1 for s in self.label_segments if s['label']=='RIGHT')} RIGHT)")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _load_epochs(
        self, path: str, apply_ica: bool
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Load, preprocess, and epoch a GDF file.
        Returns two lists of numpy arrays (epochs_left, epochs_right),
        each entry shape (n_channels, epoch_samples).
        """
        raw = mne.io.read_raw_gdf(path, preload=True, verbose=False)

        # Channel renaming
        raw.rename_channels(bci2a_strip)
        present  = set(raw.ch_names)
        safe_map = {k: v for k, v in BCI_CHANNEL_MAPPING.items() if k in present}
        if safe_map:
            raw.rename_channels(safe_map)

        raw.set_channel_types({"EOG-left": "eog", "EOG-central": "eog", "EOG-right": "eog"})
        raw.set_montage("standard_1020", verbose=False)
        raw.pick_channels(STREAM_CHANNELS, ordered=True)
        raw.filter(LOW_FREQ, HIGH_FREQ, verbose=False)

        if raw.info["sfreq"] != SAMPLING_RATE:
            raw.resample(SAMPLING_RATE, verbose=False)

        if apply_ica:
            print("[DemoStream] Running ICA ...")
            ica = ICA(n_components=6, random_state=42, max_iter="auto", verbose=False)
            ica.fit(raw, verbose=False)
            eog_idx, _ = ica.find_bads_eog(raw, verbose=False)
            ica.exclude = eog_idx
            ica.apply(raw, verbose=False)
            print(f"[DemoStream] ICA removed {len(eog_idx)} component(s)")

        # Epoch using the label mapping (Left=769->0, Right=770->1)
        event_id = {k: v for k, v in ANNOTATION_TO_CLASS.items() if v in ACTIVE_CLASSES}
        events, _ = mne.events_from_annotations(raw, event_id=event_id, verbose=False)
        epochs    = mne.Epochs(
            raw, events, event_id=event_id,
            tmin=0.5, tmax=4.5,
            baseline=None, preload=True, verbose=False,
        )

        X = epochs.get_data()            # (N, n_channels, epoch_samples)
        y = epochs.events[:, -1]         # class indices (0 or 1)

        epochs_left  = [X[i] for i in range(len(y)) if y[i] == 0]
        epochs_right = [X[i] for i in range(len(y)) if y[i] == 1]

        return epochs_left, epochs_right

    def _build_playlist(
        self,
        epochs_left:  list[np.ndarray],
        epochs_right: list[np.ndarray],
        n_trials:     int | None,
    ) -> list[tuple[np.ndarray, str]]:
        """Build ordered list of (epoch_data, label) based on mode."""
        if self.mode == "left":
            items = [(ep, "LEFT")  for ep in epochs_left]
        elif self.mode == "right":
            items = [(ep, "RIGHT") for ep in epochs_right]
        elif self.mode == "alternating":
            items = []
            for l_ep, r_ep in zip(epochs_left, epochs_right):
                items.append((l_ep, "LEFT"))
                items.append((r_ep, "RIGHT"))
        else:  # 'all' — interleave in dataset order
            items = [(ep, "LEFT")  for ep in epochs_left] + \
                    [(ep, "RIGHT") for ep in epochs_right]

        if n_trials is not None:
            items = items[:n_trials]

        if not items:
            raise ValueError(f"mode='{self.mode}' produced zero trials. "
                             "Check that the GDF file contains the required labels.")
        return items

    def _stitch(
        self, playlist: list[tuple[np.ndarray, str]]
    ) -> tuple[np.ndarray, list[dict]]:
        """
        Stitch epochs into a single continuous buffer with rest pads.

        Returns
        -------
        data          : np.ndarray  (n_channels, total_samples)
        label_segs    : list of {'start': int, 'end': int, 'label': str}
        """
        n_channels   = playlist[0][0].shape[0]
        rest_pad     = np.zeros((n_channels, self.rest_samples), dtype=np.float32)
        blocks       = [rest_pad.copy()]   # start with silence
        label_segs   = []
        cursor       = self.rest_samples

        for epoch, label in playlist:
            epoch = epoch.astype(np.float32)
            epoch_len = epoch.shape[1]

            label_segs.append({
                "start": cursor,
                "end":   cursor + epoch_len,
                "label": label,
            })

            blocks.append(epoch)
            blocks.append(rest_pad.copy())
            cursor += epoch_len + self.rest_samples

        data = np.concatenate(blocks, axis=1)  # (n_channels, total_samples)
        return data, label_segs

    # ── Public API ────────────────────────────────────────────────────────────

    def read(self, n_samples: int) -> np.ndarray:
        """Read next n_samples, looping at end."""
        end = self.pointer + n_samples
        if end <= self.n_samples:
            chunk = self.data[:, self.pointer:end].copy()
        else:
            tail      = self.data[:, self.pointer:]
            from_head = self.data[:, :end - self.n_samples]
            chunk     = np.concatenate([tail, from_head], axis=1)

        self.pointer = end % self.n_samples
        return chunk

    def get_true_label(self, window_start_sample: int, overlap_threshold: float = 0.5) -> str | None:
        """
        Return the ground-truth label when >=overlap_threshold of the window
        falls inside a labeled trial segment.

        Because the stream is built from clean, pre-extracted epochs,
        every sample inside a labeled segment has a known class.
        """
        window_end = window_start_sample + WINDOW_SAMPLES

        for seg in self.label_segments:
            overlap = max(0, min(window_end, seg["end"]) - max(window_start_sample, seg["start"]))
            if overlap >= WINDOW_SAMPLES * overlap_threshold:
                return seg["label"]

        return None

    def get_trial_info(self, window_start_sample: int) -> dict | None:
        """Return full segment info for the window, or None if in rest."""
        window_end = window_start_sample + WINDOW_SAMPLES
        for i, seg in enumerate(self.label_segments):
            overlap = max(0, min(window_end, seg["end"]) - max(window_start_sample, seg["start"]))
            if overlap >= WINDOW_SAMPLES * 0.5:
                return {**seg, "trial_idx": i + 1, "total": len(self.label_segments)}
        return None

    @property
    def current_position(self) -> int:
        return self.pointer
