# File Paths
BCI2A_PATH = './dataset/BCI-2A/'
BCI2B_PATH = './dataset/BCI-2B/'
MI_DATA_PATH = './dataset/MI2Class/'
HGD_DATA_PATH = './dataset/HGD/'

SAMPLING_RATE = 250 
LOW_FREQ = 4
HIGH_FREQ = 40
BATCH_SIZE = 32

BCI_MAPPING = {
    
    'Fz': 'Fz',
    '0': 'FC3',
    '1': 'FC1',
    '2': 'FCz',
    '3': 'FC2',
    '4': 'FC4',

    '5': 'C5',
    'C3': 'C3',
    '6': 'C1',
    'Cz': 'Cz',
    '7': 'C2',    
    'C4': 'C4',
    '8': 'C6',
      
    '9': 'CP3',
    '10': 'CP1',
    '11': 'CPz',
    '12': 'CP2',
    '13': 'CP4',

    '14': 'P1',
    'Pz': 'Pz',
    '15': 'P2',
    '16': 'POz'
}

BCI_CHANNELS = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz']