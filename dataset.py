import mne
from config import BCI2A_PATH, MI_DATA_PATH, SAMPLING_RATE, LOW_FREQ, HIGH_FREQ
import os

print(f"Checking if BCI-2A dataset path exists: {os.path.exists(BCI2A_PATH)}")
print(os.listdir(BCI2A_PATH))

def load_bci_data(path = BCI2A_PATH):
    # Load the BCI-2A dataset
    parent_dir = os.path.dirname(path)
    raw_files = [os.path.join(parent_dir, f) for f in os.listdir(parent_dir) if f.endswith('.gdf')]
    print(f"Found {len(raw_files)} raw data files in {path}.")
    raw_data = []
    for file in raw_files:
        raw = mne.io.read_raw_gdf(file, preload=True)
        raw.filter(LOW_FREQ, HIGH_FREQ)
        raw.resample(SAMPLING_RATE)
        raw_data.append(raw)
    return raw_data    

dataset = load_bci_data()
print(f"Loaded {len(dataset)} raw data files from BCI-2A dataset.")


def load_raw_mi_data(path = MI_DATA_PATH):
    # Load the MI-2Class dataset
    parent_dir = os.path.dirname(path)
    raw_files = [os.path.join(parent_dir, f) for f in os.listdir(parent_dir) if f.endswith('.fif')]
    print(f"Found {len(raw_files)} raw data files in {path}.")
    raw_data = []
    for file in raw_files:
        raw = mne.io.read_raw_fif(file, preload=True)
        raw.filter(LOW_FREQ, HIGH_FREQ)
        raw.resample(SAMPLING_RATE)
        raw_data.append(raw)
    return raw_data


