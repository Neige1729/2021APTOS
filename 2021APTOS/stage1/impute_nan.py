import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

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

csv_data_path = "data/TrainingAnnotation.csv"
batch_size = 128


def read_data():
    myDf = pd.read_csv(csv_data_path)
    df1 = myDf.drop(["patient ID"], axis=1)
    my_imputer = SimpleImputer()
    data_imputed = my_imputer.fit_transform(df1)
    df_data_imputed = pd.DataFrame(data_imputed, columns=df1.columns)
    df_data_imputed["patient ID"] = myDf["patient ID"]
    df_data_imputed = df_data_imputed.reindex(columns=myDf.columns)
    print(df_data_imputed)
    df_data_imputed.to_csv("TrainingImputed.csv")
    return df_data_imputed


if __name__ == '__main__':
    read_data()
