#coding:utf-8
#使用lstm 老办法 ，placeholder那种

# TARGET_PATH = "../../data/Train&Test/C1S4_newlabel_hybrid_timestep3/"#竖着拼四个
# TRAIN_FILE_NAME="hybrid_train_set.csv"
# VALID_FILE_NAME="hybrid_validate_set.csv"
TARGET_PATH = "../../data/Train&Test/C4S1_newlabel/"#横着拼4个
TRAIN_FILE_NAME="full_train_set.csv"
VALID_FILE_NAME="full_validate_set.csv"
TRAIN_FILE=TARGET_PATH+TRAIN_FILE_NAME
VALID_FILE= TARGET_PATH+VALID_FILE_NAME
NEW_LABEL2_PATH="../../data/Common/New_Label2/"

import warnings

warnings.filterwarnings('ignore')
import functools
import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score, roc_auc_score
from keras.models import Sequential,load_model
from keras.layers import Dense,Dropout
from keras.layers import LSTM
import feature_selection as fs
from sklearn.decomposition import PCA

# #####################################################
#处理财务指标，处理成[batch,time_step,input_size]形状
#每个股票，end_date1-4之间4个取出来

f1=open(TRAIN_FILE)
traindata=pd.read_csv(f1)
traindata=traindata.fillna(0)
train_y=traindata["label"]
train_y=np.array(train_y).reshape(-1,1)
colnames = traindata.columns.values.tolist()#所有列名
time_step1_col=colnames[colnames.index("report_type_1"):colnames.index("ann_date_2")]
time_step2_col=colnames[colnames.index("report_type_2"):colnames.index("ann_date_3")]
time_step3_col=colnames[colnames.index("report_type_3"):colnames.index("ann_date_4")]
time_step4_col=colnames[colnames.index("report_type_4"):colnames.index("label")]
train_x=[]
scaler = StandardScaler()
for index,row in traindata.iterrows():
    trax=[]#一只股票的数据[4*382]
    trax.append(row[time_step1_col])
    trax.append(row[time_step2_col])
    trax.append(row[time_step3_col])
    trax.append(row[time_step4_col])
    trax = scaler.fit_transform(trax)
    train_x.append(trax)
train_x = np.array(train_x).astype(np.float32)#[1477,4,382]
#####处理验证集
f2=open(VALID_FILE)
validdata=pd.read_csv(f2)
validdata=validdata.fillna(0)
valid_y=validdata["label"]
print(sum(valid_y==1))
valid_y=np.array(valid_y).reshape(-1,1)
print(sum(valid_y==1))
valid_x=[]
scaler = StandardScaler()
for index,row in validdata.iterrows():
    trax=[]#一只股票的数据[4*382]
    trax.append(row[time_step1_col])
    trax.append(row[time_step2_col])
    trax.append(row[time_step3_col])
    trax.append(row[time_step4_col])
    trax = scaler.fit_transform(trax)
    valid_x.append(trax)
valid_x = np.array(valid_x).astype(np.float32)#[1477,4,382]
#
# ##############################################################
# #进行pca降维
pca_threshold=10
train_x=train_x.reshape(-1,382)
pca = PCA(n_components=pca_threshold, whiten=True)
train_x = pca.fit(train_x).transform(train_x)
train_x=train_x.reshape(-1,4,pca_threshold)

valid_x=valid_x.reshape(-1,382)
pca = PCA(n_components=pca_threshold, whiten=True)
valid_x = pca.fit(valid_x).transform(valid_x)
valid_x=valid_x.reshape(-1,4,pca_threshold)

# a=1
# # train_x, train_y, valid_x, valid_y = fs.pca_method(train_x, train_y, valid_x, valid_y, pca_threshold=10, is_auto=0,is_split=0)








#############################################################
# #这里处理数据集是行情数据，hybrid的
# # 处理训练集
# f1 = open(TRAIN_FILE)
# traindata = pd.read_csv(f1)
# train_y = traindata["label"]
# train_y=np.array(train_y).reshape(-1,1)
# traindata = traindata.ix[:, -9:]
# # print(traindata.info())
# # 构造[batch_size,time_len,input_size]形状的input tensor
# train_x = []
# scaler = StandardScaler()
# for index, row in traindata.iterrows():
#     colnames = traindata.columns.values.tolist()
#     trax = []
#     for colname in colnames:
#         each = row[colname]
#         each = each[1:-1].split(",")
#         each = [float(x) for x in each]
#         trax.append(each)
#     trax = np.array(trax).T
#     trax = scaler.fit_transform(trax)  # 特征归一化 统一量纲
#     train_x.append(trax)
# train_x = np.array(train_x).astype(np.float32)
# a=train_x[0]
# b=len(train_y)
#
#     # 处理验证集
# f2 = open(VALID_FILE)
# validdata = pd.read_csv(f2)
# valid_y = validdata["label"]
# valid_y=np.array(valid_y).reshape(-1,1)
# validdata = validdata.ix[:, -9:]
# # 构造[batch_size,time_len,input_size]形状的input tensor
# valid_x = []
# scaler = StandardScaler()
# for index, row in validdata.iterrows():
#     colnames = validdata.columns.values.tolist()
#     trax = []
#     for colname in colnames:
#         each = row[colname]
#         each = each[1:-1].split(",")
#         each = [float(x) for x in each]
#         trax.append(each)
#     trax = np.array(trax).T
#     trax = scaler.fit_transform(trax)  # 特征归一化 统一量纲
#     valid_x.append(trax)
# valid_x = np.array(valid_x).astype(np.float32)
###############################################################################################

# #划分batch,得到batch_index
# def get_batch_index(train_x,batch_size,time_step):
#     batch_index=[]
#     for i in range(len(train_x)-time_step):
#        if i % batch_size==0:
#            batch_index.append(i)
#     batch_index.append((len(train_x)-time_step))
#     return batch_index
#
# create and fit the LSTM network
# look_back = 1
# model = Sequential()
# model.add(LSTM(100, input_shape=(4,pca_threshold),return_sequences=True))
#model.add(LSTM(100, input_shape=(4,382),return_sequences=True))
#model.add(LSTM(30, input_shape=(15,9)))
# model.add(LSTM(30,return_sequences=True))
# model.add(LSTM(5,return_sequences=False))
# model.add(LSTM(30,input_shape=(train_x.shape[1],train_x.shape[2])))
# model.add(Dropout(0.25))
# model.add(Dense(1,activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='adam')
# model.fit(train_x, train_y, epochs=100, batch_size=50, verbose=2)
# model.save("./lstm_keras_model/30_100epo_C1S4_hybrid_timestep3.h5")

model=load_model("./lstm_keras_model/100_30_5_300epo_pca10.h5")
pre_class=model.predict_classes(valid_x)
pre_value=model.predict(valid_x)#降维之后还是正常的，不加阈值0.9
#再跑一遍不降维的，加阈值0.9 看收益

# print("预测结果:",pre)
# pre=list(pre)
# for i in range(len(pre)):
#     if pre[i]>0.5:
#         pre[i]=1
#     else:
#         pre[i]=0
# # print("yuce:",pre2)
#
# print("原始预测出1的数量：", list(original_pre).count(1))
# print("改变阈值预测出1的数量：", list(pre).count(1))
# pre = np.array(pre)


####################################
#预测结果只取前10
print("预测结果:",pre_value)
# pre_value=list(pre_value)


print("原始预测出1的数量：", list(pre_class).count(1))





#############################################################################################
print(sum(valid_y==1))
confmat = classification_report(y_true=valid_y, y_pred=pre_class)
print(confmat)


precision = precision_score(valid_y,pre_class)
recall = recall_score(valid_y,pre_class)
acc=accuracy_score(valid_y,pre_class)
auc=roc_auc_score(valid_y,pre_class)
print("acc:",acc)
print('Test set f1:', f1_score(valid_y,pre_class))
print("precision:",precision)
print("auc:",auc)
################################################
#     #计算回报率
# df_pre = pd.DataFrame(pre_class, columns=["pre_y"])
# valid_set = pd.read_csv(VALID_FILE)
# valid_set_pre = pd.concat([valid_set, df_pre], axis=1)
# select_ts_code = list(valid_set_pre[valid_set_pre["pre_y"] == 1]["ts_code"])  #####选出预测为1的股票列表
# all_change = 0  #####所有选出的股票的收益值，为分子
# s_date_2_close_sum = 0  #####所有选出股票上个月的close和，为分母
# for each_stock in select_ts_code:
#     path = NEW_LABEL2_PATH + each_stock[:-3] + "_" + each_stock[-2:] + ".csv"
#     stock_df = pd.read_csv(path)
#     each_stock_change = stock_df[stock_df["jidu_date"] == 20180930]["s_change"].tolist()[0]
#     all_change += each_stock_change
#     s_date_2_close = stock_df[stock_df["jidu_date"] == 20180930]["s_date_2_close"].tolist()[0]
#     s_date_2_close_sum += s_date_2_close
# final_shouyilv = all_change / s_date_2_close_sum
# print("最终回报率：", final_shouyilv)

# print("test accuracy %g" % sess.run(,feed_dict={_X: images, y: labels, keep_prob: 1.0,
#                                                         batch_size: mnist.test.images.shape[0]}))



    #计算回报率,只取值的前十
df_pre = pd.DataFrame(pre_value, columns=["pre_y"])
valid_set = pd.read_csv(VALID_FILE)
valid_set_pre = pd.concat([valid_set, df_pre], axis=1)
valid_set_pre_sort=valid_set_pre.sort_values(by="pre_y",ascending=False)
select_ts_code = list(valid_set_pre_sort["ts_code"])[:10] #####选出预测为1的股票列表
all_change = 0  #####所有选出的股票的收益值，为分子
s_date_2_close_sum = 0  #####所有选出股票上个月的close和，为分母
for each_stock in select_ts_code:
    path = NEW_LABEL2_PATH + each_stock[:-3] + "_" + each_stock[-2:] + ".csv"
    stock_df = pd.read_csv(path)
    each_stock_change = stock_df[stock_df["jidu_date"] == 20180930]["s_change"].tolist()[0]
    all_change += each_stock_change
    s_date_2_close = stock_df[stock_df["jidu_date"] == 20180930]["s_date_2_close"].tolist()[0]
    s_date_2_close_sum += s_date_2_close
final_shouyilv = all_change / s_date_2_close_sum
print("最终回报率：", final_shouyilv)
