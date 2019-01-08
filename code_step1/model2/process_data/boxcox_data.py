#coding:utf-8
import pandas as pd
import numpy as np
# %matplotlib inline
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
from scipy import stats
from scipy.stats import norm, skew

TARGET_PATH = "../../../data/Train&Test/C1S1_oldlabel/"
TRAIN_FILE_NAME="full_train_set.csv"
TRAIN_FILE=TARGET_PATH+TRAIN_FILE_NAME
VALID_FILE_NAME="full_validate_set.csv"
VALID_FILE= TARGET_PATH+VALID_FILE_NAME


# traindata=pd.read_csv(open(TRAIN_FILE))
# newtrain=pd.read_csv(open(TARGET_PATH+"full_train_set_boxcox.csv"))

# #训练集boxcox偏态转正态
# f1=open(TRAIN_FILE)
# traindata=pd.read_csv(f1)
# numeric_feats=traindata.columns.tolist()
# dropcol=["ts_code","label","ann_date_1","f_ann_date_1","end_date_1"]
# for i in dropcol:
#     numeric_feats.remove(i)
# print(numeric_feats)
# # numeric_feats = data.dtypes[data.dtypes != "object"].index
#
# # Check the skew of all numerical features
# skewed_feats = traindata[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
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
#     traindata[feat] = boxcox1p(traindata[feat], lam)
#
# traindata.to_csv(TARGET_PATH+"full_train_set_boxcox.csv",index=False)

#验证集boxcox偏态转正态
f1=open(VALID_FILE)
validdata=pd.read_csv(f1)
numeric_feats=validdata.columns.tolist()
dropcol=["ts_code","label","ann_date_1","f_ann_date_1","end_date_1"]
for i in dropcol:
    numeric_feats.remove(i)
print(numeric_feats)
# numeric_feats = data.dtypes[data.dtypes != "object"].index

# Check the skew of all numerical features
skewed_feats = validdata[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
print(skewness.head(10))

skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    #all_data[feat] += 1
    validdata[feat] = boxcox1p(validdata[feat], lam)

validdata.to_csv(TARGET_PATH+"full_validate_set_boxcox.csv",index=False)