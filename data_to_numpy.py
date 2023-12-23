import mne
import torch
import numpy as np
from scipy.signal import butter, filtfilt
from pathlib import Path
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
CUT_OFF = 80
SAMPLE_POINT = 2500
data = mne.io.read_raw('preprocessed_data/neg/PN00-1_trail5_neg.edf')
data,time = data[:, 0:2500]

def Pearson(matrix):
    size = matrix.shape[0]
    mat = np.zeros((size,size))
    for i in (range(size)):
        for j in range(size):
            mat[i][j] = np.corrcoef(matrix[i],matrix[j])[0, 1]
    return mat
        
a = np.array([[1, 2, 1], [4, 7, 0], [13, 8, 3], [10, 1, 12]])
b = Pearson(a)
c = np.corrcoef(a,a,rowvar=False)
print(b == c)


def bandpass_filter(data, lowcut, highcut, fs):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    order = 5
    
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data)
    
    return filtered_data
    





def dtf(x):
    """
    Calculate directed transfer function (DTF).
    """
    n = x.shape[1]
    f = np.fft.fft(x, axis=1)  # Fourier transform
    ff = np.conj(np.swapaxes(f, -1, -2))
    g = np.linalg.solve(f @ ff / n, f)
    return g @ ff / n
def threshold_matrix(matrix, threshold=0.03):
    thresholded_matrix = np.where(matrix > threshold, 1, 0)
    return thresholded_matrix
dir = 'preprocessed_data'
for signal in tqdm(sorted(Path(dir).rglob('*.edf')),colour='cyan'):
    if signal.parts[-2] == 'pos' and  int(signal.parts[-1][2:4]) <= 10:
        #========================================#
        #    第一步：输出截至频率个数个因果传递矩阵    #
        #========================================#
        data = mne.io.read_raw(str(signal))
        data,time = data[:, 0:2500]
        # print(data[0])
        # plt.plot(data[0])
        # plt.show()
        # Set filter parameters
        fig, axes = plt.subplots(nrows=8, ncols=10, figsize=(20, 16))
        matrix_list = []
        Delta_list = []
        Theta_list = []
        Alpha_list = []
        Beta_list = []
        Gamma_list = []


        for i in range(0, CUT_OFF):
            low_f = i
            high_f = i+1
            if low_f == 0:
                low_f = 0.0001

            b, a = butter(4, [low_f/(SAMPLE_POINT/2), high_f /
                        (SAMPLE_POINT/2)], btype='band')
            filtered_data = filtfilt(b, a, data, axis=1)
            dtf_matrix = dtf(filtered_data)
            dtf_matrix = np.abs(dtf_matrix)
            if low_f in range(1, 3):
                Delta_list.append(dtf_matrix)
            if low_f in range(4, 7):
                Theta_list.append(dtf_matrix)
            if low_f in range(8, 13):
                Alpha_list.append(dtf_matrix)
            if low_f in range(14, 30):
                Beta_list.append(dtf_matrix)
            if low_f in range(30, 80):
                Gamma_list.append(dtf_matrix)

            matrix_list.append(dtf_matrix)

        # for k, ax in enumerate(axes.flat):
        #     ax.imshow(np.log10(matrix_list[k]), cmap='plasma')
        #     ax.set_axis_off()
        # plt.tight_layout()
        # plt.show()
        
        #=====================================================#
        #           指定五个特殊频段，加权叠加，得到五张频谱图       #
        #=====================================================#
        Delta_array = np.asarray(Delta_list)
        Theta_array = np.asarray(Theta_list)
        Alpha_array = np.asarray(Alpha_list)
        Beta_array = np.asarray(Beta_list)
        Gamma_array = np.asarray(Gamma_list)

        Delta_array_average = np.mean(Delta_array,axis=0)
        Theta_array_average = np.mean(Theta_array,axis=0)
        Alpha_array_average = np.mean(Alpha_array,axis=0)
        Beta_array_average = np.mean(Beta_array,axis=0)
        Gamma_array_average = np.mean(Gamma_array,axis=0)

        average_pic = [Delta_array_average,Theta_array_average,Alpha_array_average,Beta_array_average,Gamma_array_average]
        fig,axes = plt.subplots(5,figsize = (20,10))
        # for j,ax in enumerate(axes.flat):
        #     ax.imshow(np.log10(average_pic[j]), cmap='plasma')
        #     ax.set_axis_off()
        # plt.tight_layout()
        # plt.show()
        print(Delta_array_average)
        
        #=====================================================#
        #                      指定阈值                        #
        #=====================================================#
        matrices = [Delta_array_average,Theta_array_average,Alpha_array_average,Beta_array_average,Gamma_array_average]
        array = np.concatenate(matrices, axis=0)
        # array = threshold_matrix(array, threshold=0.03)
        if signal.parts[-2] == 'pos':
            np.save(f'data_numpy/pos/pos_{signal.parts[-1][0:14]}.npy',array)
        if signal.parts[-2] == 'neg':
            np.save(f'data_numpy/neg/neg_{signal.parts[-1][0:14]}.npy',array)