#coding:utf-8
#用逻辑回归进行二分类，调整验证集使其与训练集同分布
#训练集采用20180331的财报行为得到20186月的涨跌
#验证集采用20180630的财报行为得到20189月的涨跌

#TARGET_PATH = "../../data/Train&Test/C1S1_oldlabel/"
#TARGET_PATH = "../../data/Train&Test/C1S1_newlabel/"
TARGET_PATH = "../../data/Train&Test/C2S1_newlabel/"
#TARGET_PATH = "../../data/Train&Test/C1S4_newlabel/"
#TARGET_PATH = "../../data/Train&Test/C1S2_newlabel/"
TRAIN_FILE_NAME="full_train_set.csv"
TRAIN_FILE=TARGET_PATH+TRAIN_FILE_NAME
VALID_FILE_NAME="full_validate_set.csv"
VALID_FILE= TARGET_PATH+VALID_FILE_NAME

import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,f1_score,accuracy_score,roc_auc_score
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.externals import joblib
import feature_selection as fs

#
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')


def processdata():
    #训练集数据处理
    f1=open(TRAIN_FILE)
    traindata=pd.read_csv(f1)
    #对ts_code进行哈希编码和one-hot
    ts_code_30 = traindata["ts_code"].map(lambda x: (hash(x)) % 30)
    ts_code_30 = pd.get_dummies(ts_code_30, prefix="ts_code_30")
    traindata = pd.concat([ts_code_30, traindata], axis=1)
    #计数正负样本比例
    cunt = traindata[traindata.label == 1]["ts_code"]#192正例，共1710
    train_zhengnum=cunt.shape[0]
    train_funum=traindata.shape[0]-train_zhengnum
    # 加以区分SH和SZ的股票
    traindata["type"] = traindata["ts_code"].map(lambda x: 'SZ' if 'SZ' in x else 'SH')
    stocktype = pd.get_dummies(traindata["type"], prefix="type")
    traindata = traindata.drop(["type"], axis=1)
    traindata = pd.concat([traindata, stocktype], axis=1, join="outer")
    #提取x，y
    train_y=pd.DataFrame(traindata["label"])
    dropcol=["ts_code","label","ann_date_1","f_ann_date_1","end_date_1"]
    train_x=traindata.drop(dropcol,axis=1)
    #缺失值填充，数据标准化、缩放
    train_x = train_x.fillna(0)
    colnames = train_x.columns.values.tolist()
    scaler = StandardScaler()
    for colname in colnames:
        if "ts_code" in colname:
            pass
        else:
            train_x[colname]=scaler.fit_transform(np.array(train_x[colname]).reshape(-1,1))

    #验证集数据处理
    f2 = open(VALID_FILE)
    validdata = pd.read_csv(f2)
    #ts_code哈希和one-hot
    ts_code_30 = validdata["ts_code"].map(lambda x: (hash(x)) % 30)
    ts_code_30 = pd.get_dummies(ts_code_30, prefix="ts_code_30")
    validdata = pd.concat([ts_code_30, validdata], axis=1)
    #计数正负比例
    val_zheng=validdata[validdata.label==1]
    val_fu=validdata[validdata.label==0]
    cunt2 = validdata[validdata.label == 1]["ts_code"]#481正例
    #调整验证集的分布
    # val_zhengnum=cunt2.shape[0]
    # val_funum=validdata.shape[0]-val_zhengnum
    # val_new_zheng=int(val_funum*(train_zhengnum/train_funum))
    # select_val_zheng=val_zheng.sample(n=val_new_zheng)
    # validdata=pd.concat([select_val_zheng,val_fu],axis=0).reset_index(drop=True)
    # 加以区分SH和SZ的股票
    validdata["type"] = validdata["ts_code"].map(lambda x: 'SZ' if 'SZ' in x else 'SH')
    stocktype = pd.get_dummies(validdata["type"], prefix="type")
    validdata = validdata.drop(["type"], axis=1)
    validdata = pd.concat([validdata, stocktype], axis=1, join="outer")
    # 提取x，y
    valid_y = pd.DataFrame(validdata["label"])
    dropcol = ["ts_code","label", "ann_date_1", "f_ann_date_1", "end_date_1"]
    valid_x = validdata.drop(dropcol, axis=1)
    # 缺失值填充，数据标准化、缩放
    valid_x = valid_x.fillna(0)
    colnames = valid_x.columns.values.tolist()
    scaler = StandardScaler()
    for colname in colnames:
        if "ts_code" in colname:
            pass
        else:
            valid_x[colname] = scaler.fit_transform(np.array(valid_x[colname]).reshape(-1, 1))

    return train_x,train_y,valid_x,valid_y






if __name__ == '__main__':
    train_x, train_y, valid_x, valid_y=processdata()

    ####################################################
    #pca降维
    #train_x, train_y, valid_x, valid_y=fs.pca_method(train_x,train_y,valid_x,valid_y,pca_threshold=10,is_auto=0,is_split=1)
    train_x, train_y, valid_x, valid_y = fs.factor_analysis_method(train_x, train_y, valid_x, valid_y, fa_threshold=10,is_split=1)
    #train_x, train_y, valid_x, valid_y = fs.chi_method(train_x, train_y, valid_x, valid_y, chi_threshold=10, is_split=1)
    #降维之后再归一化
    scaler = MinMaxScaler()
    train_x = scaler.fit_transform(train_x)
    valid_x = scaler.fit_transform(valid_x)
#########################################################################
    model = LogisticRegression(penalty="l2")  ###样本失衡，正例样本远比负例样本少，每个分类的权重与该分类在样品中出现的频率成反比。
    ###sklearn里面逻辑回归默认是用L2正则化项的，我这个模型里面特征很多，数据集很小，所以需要l1正则化
    model.fit(train_x, train_y)
    # #保存模型
    # joblib.dump(model, "./LR_model/model_balanced.m")
    # model=joblib.load("./LR_model/model_balanced.m")
    train_y_pre = model.predict(train_x)
    print(train_y_pre)
    trainconfmat = classification_report(y_true=train_y, y_pred=train_y_pre)
    print(trainconfmat)
    print("train_acc:", model.score(train_x, train_y))
    print("train_f1:", f1_score(y_true=train_y, y_pred=train_y_pre))
    print("train_auc:", roc_auc_score(train_y, train_y_pre))

    valid_pre = model.predict(valid_x)
    print("预测出1的数量：", list(valid_pre).count(1))
    confmat = classification_report(y_true=valid_y, y_pred=valid_pre)
    f1 = f1_score(y_true=valid_y, y_pred=valid_pre)
    auc = roc_auc_score(valid_y, valid_pre)
    print(confmat)
    print("f1_score:", f1)
    print("auc_score:", auc)
    print("accuarancy:", accuracy_score(valid_y, valid_pre))

