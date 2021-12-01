import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import lagrange

data = pd.read_csv('data/TrainingAnnotation.csv', dtype={"patient ID": str})
# df = pd.DataFrame(file)

# pandas设置最大显示行和列
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 300)

# 调整显示宽度，以便整行显示
pd.set_option('display.width', 1000)

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)

# print(data)
# 重复值处理
# print(data.duplicated().value_counts())
# 缺失值处理
# print(data[data.isnull().values==True])
# data1 = data[data.isnull().values==True]
# from scipy.interpolate import lagrange
#
# def ployinterp_column(s, n, k=5):
#     y = s[list(range(n - k, n)) + list(range(n + 1, n + 1 + k))]  # 取数
#     y = y[y.notnull()]  # 剔除空值
#     return lagrange(y.index, list(y))(n)  # 插值并返回插值结果
#
# # 逐个元素判断是否需要插值
# for i in data1.columns:
#     for j in range(len(data1)):
#         if (data1[i].isnull())[j]:  # 如果为空即插值
#             data1[i][j] = ployinterp_column(data1[i], j)
#
# print(data1)
# 异常值处理
# print(data.describe())
# 散点图筛查
# plt.scatter(data["preCST"], data["CST"])
# plt.show()
# 箱型图筛查
# plt.subplot(1,2,1)
# plt.boxplot(data["preCST"])
# plt.subplot(1,2,2)
# plt.boxplot(data["CST"])
# plt.show()
print(data)