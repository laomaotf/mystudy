#! encoding=utf-8
import pandas as pd
import os,sys
import numpy as np
import random
import json

def load_data(files):
    df_all = []
    for one in files:
        df = pd.read_csv(one)
        df_all.append(df)
    return pd.concat(df_all)


df = load_data(['raw/data.csv'])
print(df.info()) #显示一些统计信息

#null处理
print("=========包含NaN的列===============")
print(df.isnull().any()) 
print("=========包含NaN的列的元素===============")
print(df[df.isnull().values==True]) 
print("=========删除包含NaN的列========")
df = df.dropna(axis=1,how="any")
print("shape:",df.shape) 

print("=====重复数据处理=====")
df.drop_duplicates(keep=False,inplace=True)
print("shape:",df.shape)

print("======删除无用的列=====")
df.drop(axis=1,columns='Id',inplace=True)
names = sorted(df.columns.values)
print("shape:",df.shape)

print("====字符串转int64====")
def replace_str2int(df, column):
    values = sorted(list(set(df[column].values)))
    valmap = dict([[val,k] for k, val in enumerate(values)])
    print(column, len(valmap))
    return df[column].map(valmap)

for name in names:
    if df[name].dtype != object:
        continue
    df[name] = replace_str2int(df,name)

#拆分train/test
N,_ = df.shape
indices_total = [k for k in range(N)]
random.shuffle(indices_total)
train_split = N * 8 // 10
indices_train,indices_test = indices_total[0:train_split],indices_total[train_split:]
df_train, df_test = df.loc[indices_train], df.loc[indices_test]
print("train : ", df_train.shape)
print("test : ", df_test.shape)

#保存
df_train.to_csv("train.csv",index=False)
df_test.to_csv("test.csv",index=False)