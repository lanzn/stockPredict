# TODO:图标签太密集
# TODO:特征有分类型，有浮点型，要分开处理
# TODO:缺失值填充，归一化以后，onehot前，填充众数或者均值

# TODO:分类型数据onehot以后，列数较多，可以单独对分类型数据降维
# TODO:连续型数据先归一化，再分桶离散化，再onehot，使全部特征统一离散化
# TODO:这样一来，全部特征都公平地进行特征选择

# TODO:可以把分类型数据先输入到MLP中，得到一个embedding,相当于降维了
# TODO:删除离群点

# TODO:数据偏态分布（skewed）了，导致离群点是分散的，而且AUC高，F1低。但是box-cox以后并未好转？
# TODO:PCA，因子分析，关联性，GA，随机森林，其他方法并结合。

# TODO:特征工程顺序---数据清洗，（离群点），正态化，标准化，归一化，特征选择
# PCA降维时，既有onehot特征，又有连续型特征，可以吗？尝试了先把onehot出来的先分出来，剩余特征降维之后再拼上的方案。


import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, roc_auc_score, accuracy_score
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

FULL_TRAIN_TEST_DATASET_PATH = "../../data/Train&Test/full_train_set.csv"


def process_data():
    # 读取dataset
    data = pd.read_csv(FULL_TRAIN_TEST_DATASET_PATH)

    # 增加"type"特征列
    data["type"] = data["ts_code"].map(lambda x: 'SZ' if 'SZ' in x else 'SH')
    stocktype = pd.get_dummies(data["type"], prefix="type")
    data = data.drop(["type"], axis=1)
    data = pd.concat([data, stocktype], axis=1, join="outer")

    # 将"ts_code"列哈希到30种以后拼接到原数据集上
    ts_code_30 = data["ts_code"].map(lambda x: (hash(x)) % 30)
    ts_code_30 = pd.get_dummies(ts_code_30, prefix="ts_code_30")
    data = pd.concat([ts_code_30, data], axis=1)

    # 对"end_date_1"进行onehot
    end_date_1 = pd.get_dummies(data["end_date_1"], prefix="end_date")
    data = data.drop(["end_date_1"], axis=1)
    data = pd.concat([data, end_date_1], axis=1, join="outer")

    # 删除没用的列
    dropcol = ["ts_code", "label", "ann_date_1", "ann_date_2", "end_date_2", "ann_date_3", "end_date_3", "ann_date_4",
               "end_date_4"]
    data_x = data.drop(dropcol, axis=1)
    data_y = pd.DataFrame(data["label"])
    return data_x, data_y, data


def pca_method(data_x, data_y, feat_labels, pca_threshold, is_auto=1):
    # 缺失值填充
    # data_x = data_x.fillna(data_x.mean())
    data_x = data_x.fillna(0)
    data_x = data_x.values
    # 归一化，之前必须保证没有空值，之后自动变成ndarray
    scaler = MinMaxScaler()
    data_x = scaler.fit_transform(data_x)
    # dataframe变成没有标签的ndarray，以便可以输入模型
    data_y = data_y.values

    # #先把onehot列单独拿出来
    # onehot_data_x_left = data_x[:, :30]
    # data_x_mid = data_x[:, 30:454]
    # onehot_data_x_right = data_x[:, 454:]

    # PCA
    if is_auto == 1:
        pca = PCA(n_components='mle', whiten=False)
    else:
        pca = PCA(n_components=pca_threshold, whiten=False)
    pca_data_x = pca.fit(data_x).transform(data_x)
    print(pca.explained_variance_ratio_)
    # data_x = np.hstack((onehot_data_x_left, pca_data_x))
    # data_x = np.hstack((data_x, onehot_data_x_right))
    return pca_data_x, data_y

def pca_method_SVM(data_x, data_y, feat_labels, pca_threshold, is_auto=1):
    # 缺失值填充
    # data_x = data_x.fillna(data_x.mean())
    data_x = data_x.fillna(0)
    data_x = data_x.values
    # 归一化，之前必须保证没有空值，之后自动变成ndarray
    scaler = MinMaxScaler()
    data_x = scaler.fit_transform(data_x)
    # dataframe变成没有标签的ndarray，以便可以输入模型
    data_y = data_y.values

    #先把onehot列单独拿出来
    onehot_data_x_left = data_x[:, :30]
    data_x_mid = data_x[:, 30:454]
    onehot_data_x_right = data_x[:, 454:]

    # PCA
    if is_auto == 1:
        pca = PCA(n_components='mle', whiten=False)
    else:
        pca = PCA(n_components=pca_threshold, whiten=False)
    pca_data_x = pca.fit(data_x_mid).transform(data_x_mid)
    print(pca.explained_variance_ratio_)
    data_x = np.hstack((onehot_data_x_left, pca_data_x))
    data_x = np.hstack((data_x, onehot_data_x_right))
    return data_x, data_y


def correlation_method(data_x, data_y, feat_labels):
    importance_dict = {}
    return importance_dict


def factor_analysis_method(data_x, data_y, feat_labels):
    importance_dict = {}
    return importance_dict


def ga_method(data_x, data_y, feat_labels):
    importance_dict = {}
    return importance_dict


def random_forest_method(data_x, data_y, feat_labels):
    # 缺失值填充
    # data_x = data_x.fillna(data_x.mean())
    data_x = data_x.fillna(0)
    data_x = data_x.values
    # 归一化，之前必须保证没有空值，之后自动变成ndarray
    scaler = MinMaxScaler()
    data_x = scaler.fit_transform(data_x)
    # dataframe变成没有标签的ndarray，以便可以输入模型
    data_y = data_y.values
    # 分割数据集
    train_x, valid_x, train_y, valid_y = train_test_split(data_x, data_y, test_size=0.25, random_state=100)

    importance_dict = {}
    forest = RandomForestClassifier(n_estimators=10, n_jobs=-1, random_state=0)
    forest.fit(train_x, train_y)
    feat_importances = forest.feature_importances_
    indices = np.argsort(feat_importances)[::-1]
    for f in range(train_x.shape[1]):
        # 给予10000颗决策树平均不纯度衰减的计算来评估特征重要性
        now_label_index = indices[f]
        print("%2d) %-*s %f" % (f+1, 30, feat_labels[now_label_index], feat_importances[now_label_index]))
        importance_dict[feat_labels[now_label_index]] = feat_importances[now_label_index]

    # 可视化特征重要性-依据平均不纯度衰减
    plt.title('Feature Importance-RandomForest')
    plt.bar(range(train_x.shape[1]), feat_importances[indices], color='lightblue', align='center')
    plt.xticks(range(train_x.shape[1]), feat_labels, rotation=90, fontsize=3)
    plt.xlim([-1, train_x.shape[1]])
    # plt.tight_layout()
    # plt.rcParams['figure.dpi'] = 300  # 分辨率
    # plt.rcParams['figure.figsize'] = (100, 50)  # 分辨率
    plt.figure(dpi=500)
    plt.show()

    # 在这个基础上，随机森林海可以通过阈值压缩数据集
    # X_selected = forest.transform(train_x, threshold=0.015)  # 大于0.15只有三个特征
    # print(X_selected.shape)
    return importance_dict


if __name__ == '__main__':
    # 处理好x和y
    data_x, data_y, data = process_data()
    # 获取所有特征名
    feat_labels = data_x.columns.values.tolist()

    # PCA，返回的数据中，数值型列没有再次进行归一化
    data_x_pca, data_y_pca = pca_method(data_x, data_y, feat_labels, 10, is_auto=0)
    # correlation_importance_dict = correlation_method(data_x, data_y, feat_labels)
    # factor_analysis_importance_dict = factor_analysis_method(data_x, data_y, feat_labels)
    # ga_importance_dict = ga_method(data_x, data_y, feat_labels)
    # 使用随机森林进行特征选择
    # random_forest_importance_dict = random_forest_method(data_x, data_y, feat_labels)

    print("Done")
