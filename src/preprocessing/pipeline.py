"""
src/preprocessing/pipeline.py
==============================
preprocess_raw() — top-level combiner: ICA → epoching → numpy arrays.
Only extracts Left (769) and Right (770) trials, matching how the
saved checkpoint was trained.
"""

import numpy as np

import mne

from src.preprocessing.artifacts import run_ica
from src.preprocessing.epoching  import create_epochs
from src.config.settings import ANNOTATION_TO_CLASS, ACTIVE_CLASSES


# Only the two trained classes
_EVENT_ID = {
    k: v
    for k, v in ANNOTATION_TO_CLASS.items()
    if v in ACTIVE_CLASSES        # "769"→0, "770"→1
}


def preprocess_raw(
    raw: mne.io.Raw,
    n_ica_components: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    raw              : mne.io.Raw  — bandpass-filtered raw object
    n_ica_components : int

    Returns
    -------
    X : np.ndarray  shape (n_epochs, n_channels, n_times)
    y : np.ndarray  shape (n_epochs,)  — values are 0 (Left) or 1 (Right)
    """
    raw_clean = run_ica(raw, n_components=n_ica_components)
    epochs    = create_epochs(raw_clean, event_id=_EVENT_ID)
    X = epochs.get_data()           # (n_epochs, n_channels, n_times)
    y = epochs.events[:, -1]        # event codes = class indices
    return X, y
