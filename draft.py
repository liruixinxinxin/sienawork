import numpy as np

# 北京历年人口数据
bj_population = [1961, 2000, 2049, 2090, 2133, 2171, 2174, 2174, 2174, 2174]
# 成都历年人口数据
cd_population = [1402, 1421, 1439, 1455, 1473, 1492, 1521, 1544, 1565, 1588]

# 计算两组数据的 Pearson 相关系数
corr_coef = np.corrcoef(bj_population, cd_population)[0][1]

print(f"The Pearson correlation coefficient between the populations of Beijing and Chengdu is {corr_coef:.4f}")
