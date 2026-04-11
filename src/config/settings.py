from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).resolve().parents[2]
DATA_DIR   = ROOT_DIR / "dataset" / "BCI-2A"
OUTPUT_DIR = ROOT_DIR / "output"
MODEL_PATH = OUTPUT_DIR / "eeg_model_improved.pth"

# ── Signal ───────────────────────────────────────────────────────────────────
SAMPLING_RATE = 250
LOW_FREQ      = 8.0  
HIGH_FREQ     = 30.0 

# ── Channels ─────────────────────────────────────────────────────────────────
# MOTOR_CHANNELS = ["C3", "C4", "Cz", "FC3", "FC4", "CP3", "CP4"]
MOTOR_CHANNELS = ["C3", "C4", "Cz"]
EOG_CHANNELS   = ["EOG-left", "EOG-central", "EOG-right"]
STREAM_CHANNELS = MOTOR_CHANNELS + EOG_CHANNELS
N_CHANNELS      = len(STREAM_CHANNELS)   # 10

# ── Model ────────────────────────────────────────────────────────────────────
N_CLASSES_CHECKPOINT = 4  
ACTIVE_CLASSES       = [0, 1]   # Left=0, Right=1 
N_ACTIVE_CLASSES     = len(ACTIVE_CLASSES)  # 2

# ── BCI-2a Channel Renaming ───────────────────────────────────────────────
def bci2a_strip(name: str) -> str:
    return name.replace("EEG-", "")

BCI_CHANNEL_MAPPING = {
    "Fz":  "Fz",
    "0":   "FC3",  "1":  "FC1",  "2":  "FCz",  "3":  "FC2",  "4":  "FC4",
    "5":   "C5",   "C3": "C3",   "6":  "C1",   "Cz": "Cz",
    "7":   "C2",   "C4": "C4",   "8":  "C6",
    "9":   "CP3",  "10": "CP1",  "11": "CPz",  "12": "CP2",  "13": "CP4",
    "14":  "P1",   "Pz": "Pz",   "15": "P2",   "16": "POz",
}


# ── BCI-2b Channel Renaming ───────────────────────────────────────────────
def bci2b_strip(name: str) -> str:
    return name.replace("EEG:", "").replace("EOG:", "")

BCI2B_CHANNEL_MAPPING = {
    "C3": "C3",   "Cz": "Cz",
    "C4": "C4",   "ch1": "EOG-left",   
    "ch2": "EOG-central",   "ch3": "EOG-right"
}



# ── Label Mappings ────────────────────────────────────────────────────────────
ANNOTATION_TO_CLASS: dict[str, int] = {
    "769": 0,   # Left Hand
    "770": 1,   # Right Hand
    "771": 2,   # Feet
    "772": 3,   # Tongue
}

CLASS_TO_LABEL: dict[int, str] = {
    0: "LEFT",
    1: "RIGHT",
    2: "FEET",
    3: "TONGUE",
}
# ── Sliding Window ───────────────────────────────────────────────────────────
WINDOW_SIZE_SEC = 4.0                                    
STEP_SIZE_SEC   = 0.25                                   
WINDOW_SAMPLES  = int(WINDOW_SIZE_SEC * SAMPLING_RATE)   
STEP_SAMPLES    = int(STEP_SIZE_SEC   * SAMPLING_RATE)   

# ── Decision Smoothing ───────────────────────────────────────────────────────
SMOOTHER_WINDOW      = 6  
CONFIDENCE_THRESHOLD = 0.75  

# ── Control Policy ───────────────────────────────────────────────────────────
# Prevents spurious actions.
DWELL_STEPS = 3
