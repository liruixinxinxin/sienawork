import mne
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm


time_of_trail = 5

channels = ['Fp1','FP2','F3','F4','Fz','C3','C4','Cz','P3','P4','Pz','F7','F8','T3','T4','T5','T6']

dir = '/home/ruixing/workspace/brain_network/siena-scalp-eeg-database-1.0.0'
for i in tqdm(sorted(Path(dir).rglob('*.edf')),colour='cyan'):  
    print(f'正在处理{i.parts[-1]}的数据')
    data = mne.io.read_raw_edf(str(i),preload=True)
    try:
        mapping1 = {'EEG FP2':'EEG Fp2'}
        mapping2 = {'EEG CZ':'EEG Cz'}
        data = data.rename_channels(mapping1)
        data = data.rename_channels(mapping2)
        print('电极名称大小写规范完成')
    except:
        print('电极名称无误')
    mapping3 = {
                'EEG Fp1':'Fp1', 'EEG Fp2':'FP2',                      #前额
                'EEG F3':'F3','EEG F4':'F4','EEG Fz':'Fz',             #额
                'EEG C3':'C3','EEG C4':'C4','EEG Cz':'Cz',             #中央
                'EEG P3':'P3','EEG P4':'P4','EEG Pz':'Pz',             #顶 
                'EEG F7':'F7','EEG F8':'F8',                           #侧额
                'EEG T3':'T3','EEG T4':'T4',                           #颞
                'EEG T5':'T5','EEG T6':'T6'                            #后颞
                }

    print(data.info)
    # data.plot(scalings = 0.00008)
    # plt.show()
    t_min = 0
    t_max = 60
    data.crop(tmin=t_min,tmax=t_max)
    data.filter(l_freq=0.1,h_freq=80)
    data = data.notch_filter(50,notch_widths = 2)
    data.resample(500)
    print(data.ch_names)
    data = data.rename_channels(mapping3)
    data.pick_channels(channels)
    # data.plot(scalings = 0.00008)
    # plt.show()
    p = 1
    tmin = 0
    tmax = t_max - t_min
    while(1):
        one_trail_data =  data.copy().crop(tmin=tmin,tmax=tmin+5)
        mne.export.export_raw(f'preprocessed_data/neg/{i.parts[-1][0:6]}_trail{p}_neg.edf', one_trail_data, overwrite = True)
        tmin = tmin + time_of_trail
        if(tmin > (tmax-time_of_trail)):
            break
        p += 1
    print(f'{i.parts[-1]}癫痫数据处理完成')