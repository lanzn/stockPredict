#coding:utf-8
#使用SVM，用全样本，对ts_code进行哈希和onehot
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split,GridSearchCV

from sklearn.metrics import classification_report,f1_score,roc_auc_score,accuracy_score
from sklearn.externals import joblib


def processData(f1):
    data=pd.read_csv(f1)
    data["type"] = data["ts_code"].map(lambda x: 'SZ' if 'SZ' in x else 'SH')
    ts_code_30=data["ts_code"].map(lambda x:(hash(x))%30)
    ts_code_30=pd.get_dummies(ts_code_30,prefix="ts_code_30")
    data=pd.concat([ts_code_30,data],axis=1)
    datay = pd.DataFrame(data["label"])
    dropcol = ["ts_code","label", "ann_date_1", "ann_date_2", "end_date_2", "ann_date_3", "end_date_3", "ann_date_4",
               "end_date_4"]
    datax = data.drop(dropcol, axis=1)

    #加以区分SH和SZ的股票
    stocktype=pd.get_dummies(datax["type"],prefix="type")
    datax = datax.drop(["type"], axis=1)
    datax = pd.concat([datax, stocktype], axis=1, join="outer")

    end_date_1 = pd.get_dummies(datax["end_date_1"], prefix="end_date")
    datax = datax.drop(["end_date_1"], axis=1)
    datax = pd.concat([datax, end_date_1], axis=1, join="outer")
    datax=datax.fillna(0)
    f1.close()
    return datax,datay,data

if __name__ == '__main__':
    f1 = open("../../data/Train&Test/full_train_set.csv")
    datax,datay,data=processData(f1)
    datax=np.array(datax)
    datay=np.array(datay)
    scaler = MinMaxScaler()
    datax=scaler.fit_transform(datax)

    train_x,valid_x,train_y,valid_y=train_test_split(datax,datay,test_size=0.25,random_state=100)

    # tuned_parameters = [{'C': [1, 10, 100, 1000]}]#C=1最好
    # clf = GridSearchCV(SVC(kernel='rbf',gamma=1), tuned_parameters, cv=5,scoring="accuracy")


    # clf=SVC(gamma=1,kernel="rbf",C=1)
    # clf.fit(train_x,train_y)
    # joblib.dump(clf, "./svm_model/model_qufenszsh.m")
    # print(clf.best_params_, clf.best_score_)
    clf=joblib.load("./svm_model/model_qufenszsh.m")
    valid_pre=clf.predict(valid_x)
    confmat=classification_report(y_true=valid_y,y_pred=valid_pre)
    f1=f1_score(y_true=valid_y,y_pred=valid_pre)
    auc=roc_auc_score(valid_y,valid_pre)
    print(confmat)
    print("f1_score:",f1)
    print("auc_score:",auc)
    print("accuarancy:",accuracy_score(valid_y,valid_pre))
# print(hash("000002.SZ")%30)