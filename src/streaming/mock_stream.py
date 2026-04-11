"""
src/streaming/mock_stream.py
============================
MockEEGStream — replays a BCI-2a TRAINING .gdf file as a continuous
EEG stream, including ground-truth label lookup.

Why training files?
  Evaluation files (.gdf ending in 'E') have no event annotations
  (labels are in separate .mat files).  Training files ('T') have
  both the raw signal AND full event annotations, so we can display
  the ground-truth motor-imagery class alongside each prediction.

Usage:
    stream = MockEEGStream("data/BCI-2A/A01T.gdf", apply_ica=True)
    chunk  = stream.read(n_samples=62)          # (n_channels, 62)
    label  = stream.get_true_label(win_start)   # "LEFT" | "RIGHT" | None
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
    bci2a_strip,
)


class MockEEGStream:
    """
    Wraps a preprocessed training .gdf file as a mock real-time EEG
    stream.  Preprocessing (bandpass + optional ICA) is done once at
    construction time; afterwards data is served sample-by-sample.

    Attributes
    ----------
    data       : np.ndarray  (n_channels, total_samples)
    n_channels : int
    n_samples  : int
    fs         : int
    events     : list[dict]  — [{sample, class_id, label}, ...]
    """

    def __init__(self, gdf_path: str, apply_ica: bool = True):
        print(f"[Stream] Loading: {gdf_path}")
        raw = self._load_and_prepare(gdf_path, apply_ica)

        self.data       = raw.get_data()          # (n_ch, n_samples)
        self.fs         = int(raw.info["sfreq"])
        self.n_channels = self.data.shape[0]
        self.n_samples  = self.data.shape[1]
        self.pointer    = 0

        self.events = self._extract_events(raw)
        print(
            f"[Stream] Ready — {self.n_channels} channels | "
            f"{self.n_samples} samples ({self.n_samples/self.fs:.1f} s) | "
            f"{len(self.events)} labeled trials"
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _load_and_prepare(self, path: str, apply_ica: bool) -> mne.io.Raw:
        raw = mne.io.read_raw_gdf(path, preload=True, verbose=False)

        # 1. Strip 'EEG-' prefix from channel names
        raw.rename_channels(bci2a_strip)

        # 2. Rename numeric labels → 10-20 standard names
        #    Filter the mapping to avoid KeyError for any channel not present
        present     = set(raw.ch_names)
        safe_map    = {k: v for k, v in BCI_CHANNEL_MAPPING.items() if k in present}
        if safe_map:
            raw.rename_channels(safe_map)

        # 3. Mark EOG channel types so ICA can find them
        raw.set_channel_types({
            "EOG-left":    "eog",
            "EOG-central": "eog",
            "EOG-right":   "eog",
        })
        raw.set_montage("standard_1020", verbose=False)

        # 4. Pick the 10 channels used during training
        raw.pick_channels(STREAM_CHANNELS, ordered=True)

        # 5. Band-pass filter
        raw.filter(LOW_FREQ, HIGH_FREQ, verbose=False)

        # 6. Resample if needed
        if raw.info["sfreq"] != SAMPLING_RATE:
            raw.resample(SAMPLING_RATE, verbose=False)

        # 7. ICA — remove ocular artifacts (same as training)
        if apply_ica:
            print("[Stream] Running ICA … (≈30 s first time)")
            ica = ICA(n_components=6, random_state=42, max_iter="auto", verbose=False)
            ica.fit(raw, verbose=False)
            eog_idx, _ = ica.find_bads_eog(raw, verbose=False)
            ica.exclude = eog_idx
            ica.apply(raw, verbose=False)
            print(f"[Stream] ICA removed {len(eog_idx)} component(s)")

        return raw

    def _extract_events(self, raw: mne.io.Raw) -> list[dict]:
        """
        Parse MNE annotations and return a list of dicts:
          { sample: int, class_id: int, label: str }
        Only Left (769) and Right (770) trials are kept.
        """
        events, event_id_map = mne.events_from_annotations(raw, verbose=False)

        # Invert the MNE map: int_code → annotation_string
        code_to_ann = {v: k for k, v in event_id_map.items()}

        result = []
        for sample, _, code in events:
            ann_str = code_to_ann.get(int(code), "")
            if ann_str in ANNOTATION_TO_CLASS:
                class_id = ANNOTATION_TO_CLASS[ann_str]
                if class_id in (0, 1):          # Only Left / Right for this demo
                    result.append({
                        "sample":   int(sample),
                        "class_id": class_id,
                        "label":    CLASS_TO_LABEL[class_id],
                    })
        return result

    # ── Public API ────────────────────────────────────────────────────────────

    def read(self, n_samples: int) -> np.ndarray:
        """
        Return the next *n_samples* from the stream.
        Seamlessly loops back to the start when the file ends.

        Returns
        -------
        chunk : np.ndarray  shape (n_channels, n_samples)
        """
        end = self.pointer + n_samples
        if end <= self.n_samples:
            chunk = self.data[:, self.pointer:end].copy()
        else:
            # Wrap around: stitch tail + head
            tail      = self.data[:, self.pointer:]
            from_head = self.data[:, : end - self.n_samples]
            chunk     = np.concatenate([tail, from_head], axis=1)

        self.pointer = end % self.n_samples
        return chunk

    def get_true_label(self, window_start_sample: int) -> str | None:
        """
        Return the ground-truth label if any annotated trial starts inside
        the window [window_start_sample, window_start_sample + WINDOW_SAMPLES).
        Returns None during inter-trial / rest periods.
        """
        window_end = window_start_sample + WINDOW_SAMPLES
        for ev in self.events:
            s = ev["sample"]
            if window_start_sample <= s < window_end:
                return ev["label"]
        return None

    @property
    def current_position(self) -> int:
        """Current read pointer (sample index in the file)."""
        return self.pointer
