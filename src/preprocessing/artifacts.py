import mne
from mne.preprocessing import ICA


def run_ica(raw: mne.io.Raw, n_components: int = 6) -> mne.io.Raw:
    """
    Fit FastICA, automatically find EOG-correlated components,
    exclude them, and return a cleaned copy of *raw*.
    """
    ica = ICA(n_components=n_components, random_state=42, max_iter="auto", verbose=False)
    ica.fit(raw, verbose=False)

    eog_indices, _ = ica.find_bads_eog(raw, verbose=False)
    ica.exclude    = eog_indices

    raw_clean = raw.copy()
    ica.apply(raw_clean, verbose=False)
    return raw_clean
