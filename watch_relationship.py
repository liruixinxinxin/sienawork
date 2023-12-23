import mne
import numpy as np
from mne.time_frequency import tfr_morlet
from mne.connectivity import spectral_connectivity
from brainplot import plot

# 定义一些变量
interictal_fname = 'preprocessed_data/neg/PN00-1_trail1_neg.edf'  # 发作间期数据文件名
ictal_fname = 'preprocessed_data/pos/PN00-1_trail1_pos.edf'  # 发作期数据文件名
sfreq = 500  # 采样率

# 读取数据并预处理
interictal_raw = mne.io.read_raw_edf(interictal_fname)
interictal_raw.set_montage('standard_1020')  # 设置电极位置信息
interictal_epochs = mne.make_fixed_length_epochs(interictal_raw, duration=5)  # 切割成2秒的Epochs

ictal_raw = mne.io.read_raw_edf(ictal_fname)
ictal_raw.set_montage('standard_1020')
ictal_epochs = mne.make_fixed_length_epochs(ictal_raw, duration=5)

# 计算Granger Causality
freqs = np.arange(1, 101, 1)  # 频率范围
n_cycles = freqs / 2.  # 每个频率的波数
interictal_power, interictal_itc = tfr_morlet(interictal_epochs, freqs=freqs, n_cycles=n_cycles, return_itc=True)  # Morlet小波变换
ictal_power, ictal_itc = tfr_morlet(ictal_epochs, freqs=freqs, n_cycles=n_cycles, return_itc=True)
interictal_connectivity, interictal_freqs, _ = spectral_connectivity(interictal_power, method='granger', mode='multitaper', sfreq=sfreq, fmin=1, fmax=80, tmin=0, tmax=5, epochs_average=False)
ictal_connectivity, ictal_freqs, _ = spectral_connectivity(ictal_power, method='granger', mode='multitaper', sfreq=sfreq, fmin=1, fmax=80, tmin=0, tmax=5, epochs_average=False)

# 将因果关系可视化
plot(interictal_connectivity, interictal_raw.info, mode='3d', title='Interictal Connectivity')
plot(ictal_connectivity, ictal_raw.info, mode='3d', title='Ictal Connectivity')