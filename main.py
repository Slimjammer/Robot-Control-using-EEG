from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import preprocess, Preprocessor
from config import BCI2A_PATH, MI_DATA_PATH, SAMPLING_RATE, LOW_FREQ, HIGH_FREQ


# Loading High Gamma Dataset by Schirrmeister et al. (2017)
dataset = MOABBDataset(dataset_name="schirrmeister2017")

# Preprocessing The Dataset
preprocessors = [
    Preprocessor("pick_types", eeg=True, meg=False, stim=False), 
    Preprocessor("resample", sfreq=SAMPLING_RATE), 
    Preprocessor("filter", l_freq=LOW_FREQ, h_freq=HIGH_FREQ),
    Preprocessor("exponential_moving_standardize", factor_new=0.001),
]

preprocess(dataset, preprocessors)

