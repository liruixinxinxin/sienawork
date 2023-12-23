from pathlib import Path
from datetime import datetime, timedelta

def time_difference(time1, time2):
    """
    计算两个时间之间的差距（单位为秒），可以处理跨天情况
    :param time1: 第一个时间，格式为"HH.MM.SS"
    :param time2: 第二个时间，格式为"HH.MM.SS"
    :return: 两个时间之间的差距（单位为秒）
    """
    t1 = datetime.strptime(time1, '%H.%M.%S')
    t2 = datetime.strptime(time2, '%H.%M.%S')
    if t1 <= t2:
        # 如果第一个时间在第二个时间之前，说明两个时间不跨天
        td = t2 - t1
    else:
        # 如果第一个时间在第二个时间之后，说明两个时间跨天
        td = timedelta(days=1) - (t1 - t2)
    return td.seconds

epilepsy_word = 'Seizure start time: '
record_word = 'Registration start time: '
e_end_word = 'Seizure end time: '
patient_num = [0,1,3,5,6,7,9,10,11,12,13,14,16,17]
for n,i in enumerate(range(len(patient_num))):
    if patient_num[n] < 10 :
        with open(f'siena-scalp-eeg-database-1.0.0/PN0{patient_num[n]}/Seizures-list-PN0{patient_num[n]}.txt', 'r') as f:
            lines = f.readlines()
    if patient_num[n] > 9 :
        with open(f'siena-scalp-eeg-database-1.0.0/PN{patient_num[n]}/Seizures-list-PN{patient_num[n]}.txt', 'r') as f:
            lines = f.readlines()

    record_results = []
    epilepsy_results = []
    e_end_result = []
    for line in lines:
        if record_word in line:
            index = line.index(record_word)
            result = line[index+len(record_word):index+len(record_word)+8]
            record_results.append(result)

    for line in lines:
        if epilepsy_word in line:
            index = line.index(epilepsy_word)
            result = line[index+len(epilepsy_word):index+len(epilepsy_word)+8]
            epilepsy_results.append(result)

    for line in lines:
        if e_end_word in line:
            index = line.index(e_end_word)
            result = line[index+len(e_end_word):index+len(e_end_word)+8]
            e_end_result.append(result)

    # print(record_results)
    # print(epilepsy_results)
    # print(e_end_result)
    time = []
    for i in range(len(record_results)):
        start_time = time_difference(record_results[i],epilepsy_results[i])
        end_time = time_difference(record_results[i],e_end_result[i])
        time.append([start_time,end_time])
    print(f'{str(patient_num[n])}:{time}')