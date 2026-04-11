"""
load_bci2a_data() — loads BCI-2a .gdf files using MNE, 
load_mi_data()    - loads MI     .fif files using MNE,
applies channel renaming, bandpass filtering, and channel selection.
"""
import os
import mne

from src.config.settings import (
    SAMPLING_RATE,
    LOW_FREQ,
    HIGH_FREQ,
    BCI_CHANNEL_MAPPING,
    STREAM_CHANNELS,
    bci2a_strip,
)


def load_bci2a_data(
    data_dir: str,
    file_suffix: str = "T.gdf",
    channels: list[str] | None = None,
    low_freq: float = LOW_FREQ,
    high_freq: float = HIGH_FREQ,
) -> list[mne.io.Raw]:
    """
    Load all BCI-2a .gdf files matching *file_suffix* from *data_dir*.

    Parameters
    ----------
    data_dir    : str   — directory containing the .gdf files
    file_suffix : str   — 'T.gdf' for training, 'E.gdf' for evaluation
    channels    : list  — channel subset to pick (None = STREAM_CHANNELS)
    """
    if channels is None:
        channels = STREAM_CHANNELS

    files = sorted(
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith(file_suffix)
    )
    print(f"[Loader] Found {len(files)} file(s) matching '*{file_suffix}'")

    raws = []
    for path in files:
        raw = mne.io.read_raw_gdf(path, preload=True, verbose=False)

        raw.rename_channels(bci2a_strip)
        present  = set(raw.ch_names)
        safe_map = {k: v for k, v in BCI_CHANNEL_MAPPING.items() if k in present}
        if safe_map:
            raw.rename_channels(safe_map)

        raw.set_channel_types({
            "EOG-left":    "eog",
            "EOG-central": "eog",
            "EOG-right":   "eog",
        })
        raw.set_montage("standard_1020", verbose=False)
        raw.pick_channels(channels, ordered=True)
        raw.filter(low_freq, high_freq, verbose=False)

        if raw.info["sfreq"] != SAMPLING_RATE:
            raw.resample(SAMPLING_RATE, verbose=False)

        raws.append(raw)
        print(f"  Loaded {os.path.basename(path)}")

    return raws


def load_mi_data(
    data_dir: str,
    file_suffix: str = ".fif",
    channels: list[str] | None = None,
    low_freq: float = LOW_FREQ,
    high_freq: float = HIGH_FREQ,
) -> list[mne.io.Raw]:
    """
    Load all MI Dataset .fif files matching *file_suffix* from *data_dir*.

    Parameters
    ----------
    data_dir    : str   — directory containing the .fif files
    file_suffix : str   — '.fif'
    channels    : list  — channel subset to pick (None = STREAM_CHANNELS)
    """
    if channels is None:
        channels = STREAM_CHANNELS

    files = sorted(
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith(file_suffix)
    )
    print(f"[Loader] Found {len(files)} file(s) matching '*{file_suffix}'")

    raws = []
    for path in files:
        raw = mne.io.read_raw_fif(path, preload=True, verbose=False)

        raw.set_montage("standard_1020", verbose=False)
        raw.pick_channels(channels, ordered=True)
        raw.filter(low_freq, high_freq, verbose=False)

        if raw.info["sfreq"] != SAMPLING_RATE:
            raw.resample(SAMPLING_RATE, verbose=False)

        raws.append(raw)
        print(f"  Loaded {os.path.basename(path)}")

    return raws