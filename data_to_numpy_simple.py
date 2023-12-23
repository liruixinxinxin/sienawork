import mne
import torch
import numpy as np
from scipy.signal import butter, filtfilt
from pathlib import Path
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
CUT_OFF = 80
SAMPLE_POINT = 2500

dir = 'preprocessed_data'
for i in tqdm(sorted(Path(dir).rglob('*.edf'))):
    data = mne.io.read_raw('preprocessed_data/neg/PN00-1_trail5_neg.edf',preload=True)
    data.filter(l_freq = 0.5,h_freq = 45)
    data,time = data[:, 0:2500]

    def Pearson(matrix):
        size = matrix.shape[0]
        mat = np.zeros((size,size))
        for i in (range(size)):
            for j in range(size):
                mat[i][j] = np.corrcoef(matrix[i],matrix[j])[0, 1]
        return mat
            
    b = Pearson(data)
    print(b)
    if i.parts[-2] == 'pos':
        np.save(f'data_numpy_simple/pos/pos_{i.parts[-1][0:14]}.npy',b)
    if i.parts[-2] == 'neg':
        np.save(f'data_numpy_simple/neg/neg_{i.parts[-1][0:14]}.npy',b)
