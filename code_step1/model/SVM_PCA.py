#coding:utf-8
#使用SVM，用全样本，对ts_code进行哈希和onehot
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split,GridSearchCV

from sklearn.metrics import classification_report,f1_score,roc_auc_score,accuracy_score
from sklearn.externals import joblib
import feature_selection as fs

if __name__ == '__main__':
    data_x, data_y, data = fs.process_data()
    # 获取所有特征名
    feat_labels = data_x.columns.values.tolist()

    # PCA，返回的数据中，数值型列没有再次进行归一化
    data_x_pca, data_y_pca = fs.pca_method_SVM(data_x, data_y, feat_labels, 40, is_auto=0)

    datax=np.array(data_x_pca)
    scaler = MinMaxScaler()
    datax=scaler.fit_transform(datax)

    train_x,valid_x,train_y,valid_y=train_test_split(datax,data_y_pca,test_size=0.25,random_state=100)

    # tuned_parameters = [{'C': [1, 10, 100, 1000]}]#C=1最好
    # clf = GridSearchCV(SVC(kernel='rbf',gamma=1), tuned_parameters, cv=5,scoring="accuracy")


    # clf=SVC(gamma=1,kernel="rbf",C=1)
    # clf.fit(train_x,train_y)
    # joblib.dump(clf, "./svm_model/model_qufenszsh_pca_40.m")
    clf=joblib.load("./svm_model/model_qufenszsh_pca_40.m")
    # print(clf.best_params_, clf.best_score_)
    valid_pre=clf.predict(valid_x)
    confmat=classification_report(y_true=valid_y,y_pred=valid_pre)
    f1=f1_score(y_true=valid_y,y_pred=valid_pre)
    auc=roc_auc_score(valid_y,valid_pre)
    print(confmat)
    print("f1_score:",f1)
    print("auc_score:",auc)
    print("accuarancy:",accuracy_score(valid_y,valid_pre))
# print(hash("000002.SZ")%30)