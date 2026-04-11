import mne
import numpy as np


def create_epochs(
    raw: mne.io.Raw,
    event_id: dict,
    tmin: float = 0.5,
    tmax: float = 4.5,
) -> mne.Epochs:
    """
    Parameters
    ----------
    raw      : mne.io.Raw  — preprocessed continuous EEG
    event_id : dict        — annotation string → class index mapping
    tmin     : float       — epoch start relative to event (s)
    tmax     : float       — epoch end   relative to event (s)

    Returns
    -------
    mne.Epochs  (preloaded)
    """
    events, _ = mne.events_from_annotations(raw, event_id=event_id, verbose=False)
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        verbose=False,
    )
    return epochs
