#coding:utf-8
#财报数据规范化，去除离群点
import pandas as pd
import numpy as np
# %matplotlib inline
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
from scipy import stats
from scipy.stats import norm, skew


def aver(df):
    for column in list(df.columns[df.isnull().sum() > 0]):
        mean_val = df[column].mean()
        df[column].fillna(mean_val, inplace=True)
    return df

def box(df):
    index={}
    dropcol = ["ts_code", "ann_date_1", "end_date_1", "ann_date_2", "end_date_2", "ann_date_3", "end_date_3",
               "ann_date_4", "end_date_4"]
    for colname in df.columns.tolist():
        if colname not in dropcol:
            Q2=df[colname].quantile()
            index[colname]=[]
            Q1=df[colname].quantile(0.25)
            Q3=df[colname].quantile(0.75)
            IQR=Q3-Q1
            min=Q1-2.5*IQR
            max=Q3+3.5*IQR
            for i in range(df.shape[0]):
                if df[colname][i]<min or df[colname][i]>max:
                    print(colname,i)
                    df.loc[i,colname]=Q2
                    index[colname].append(i)
    return df,index

def box2(df):
    index = {}
    dropcol = ["label","ts_code", "ann_date_1", "end_date_1", "ann_date_2", "end_date_2", "ann_date_3", "end_date_3",
               "ann_date_4", "end_date_4"]
    for colname in df.columns.tolist():
        if colname not in dropcol:
            max=df[colname].quantile(0.99975)
            index[colname] = []
            for i in range(df.shape[0]):
                if df[colname][i]>max:
                    print(colname,i)
                    #df.loc[i,colname]=df[colname].mean()
                    df.loc[i,colname]=0
                    index[colname].append(i)
    return df,index



pd.set_option('display.max_columns', None)

File="../../data/Train&Test/full_train_set_boxcox.csv"
data=pd.read_csv(File)

normaldata,index=box(data)
for i in index.keys():
    print(len(index[i]))

normaldata.to_csv("../../data/Train&Test/full_train_set_boxcox_normal.csv",index=False)
# data=aver(data)

# numeric_feats=data.columns.tolist()
# dropcol=["label","ts_code","ann_date_1","end_date_1","ann_date_2","end_date_2","ann_date_3","end_date_3","ann_date_4","end_date_4"]
# for i in dropcol:
#     numeric_feats.remove(i)
# print(numeric_feats)
# # numeric_feats = data.dtypes[data.dtypes != "object"].index
#
# # Check the skew of all numerical features
# skewed_feats = data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
#
# print("\nSkew in numerical features: \n")
# skewness = pd.DataFrame({'Skew' :skewed_feats})
# print(skewness.head(10))
#
# skewness = skewness[abs(skewness) > 0.75]
# print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))
#
# from scipy.special import boxcox1p
# skewed_features = skewness.index
# lam = 0.15
# for feat in skewed_features:
#     #all_data[feat] += 1
#     data[feat] = boxcox1p(data[feat], lam)
#
# data.to_csv("../../data/Train&Test/full_train_set_boxcox.csv",index=False)

#normal2只除掉0.999，填充用均值
#normal3 除掉0.9995,填充用均值
#normal4 除掉0.99975 填充用0
# normal_data,index=box2(data)
# normal_data.to_csv("../../data/Train&Test/full_train_set_normal_4.csv",index=False)
# print(index)
# for i in index.keys():
#     print(len(index[i]))

# a=1
# print(data["eps_1"].quantile(0.25))
# print(data["eps_1"][1])
#
# dropcol=["ts_code","ann_date_1","end_date_1","ann_date_2","end_date_2","ann_date_3","end_date_3","ann_date_4","end_date_4"]
# print(data.columns)
# data=data.drop(dropcol,axis=1)
# print(data.ix[:,:10].describe())
#
# col=data["total_revenue_ps_1"].tolist()
# # for i,j in enumerate(col):
# #     if j>100:
# #         print(i,j)
# print(data["fa_turn_1"][data["fa_turn_1"]>1000].count())
# a=1