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
# TODO:应该加入【利润表、负债表、现金流量表】，考虑是否用业绩快报代替现在的财务指标数据
# TODO:可以尝试不onehot，只要特征值之间的距离计算计算得合理，那么使用label encoding也没有问题。

# TODO:改成适应现在新数据的

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2
from sklearn.decomposition import FactorAnalysis
from minepy import MINE
from scipy.stats import pearsonr

from sklearn.metrics import classification_report, f1_score, roc_auc_score, accuracy_score
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

TARGET_PATH = "new_test/"
TRAIN_DATASET_PATH = "../../data/Train&Test/" + TARGET_PATH + "full_train_set.csv"
VALIDATE_DATASET_PATH = "../../data/Train&Test/" + TARGET_PATH + "full_validate_set.csv"
DROP_COLS = ["ts_code", "ann_date_1", "f_ann_date_1", "end_date_1", "label"]


def process_data():
    # 读取dataset
    train_data = pd.read_csv(TRAIN_DATASET_PATH)
    validate_data = pd.read_csv(VALIDATE_DATASET_PATH)
    split_index = train_data.shape[0]
    full_data = pd.concat([train_data, validate_data], axis=0)
    full_data = full_data.reset_index(drop=True)
    # TODO:获取trainset和validateset的边界，并将他们纵向连接在一起，参与后续处理，处理后再拆开

    # 增加"type"特征列
    # train_data["type"] = train_data["ts_code"].map(lambda x: 'SZ' if 'SZ' in x else 'SH')
    # stocktype = pd.get_dummies(train_data["type"], prefix="type")
    # train_data = train_data.drop(["type"], axis=1)
    # train_data = pd.concat([train_data, stocktype], axis=1, join="outer")

    # 将"ts_code"列哈希到30种以后拼接到原数据集上
    ts_code_30 = full_data["ts_code"].map(lambda x: (hash(x)) % 30)
    ts_code_30 = pd.get_dummies(ts_code_30, prefix="ts_code_30")
    full_data = pd.concat([ts_code_30, full_data], axis=1)

    # 对"end_date_1"进行onehot
    # end_date_1 = pd.get_dummies(train_data["end_date_1"], prefix="end_date")
    # train_data = train_data.drop(["end_date_1"], axis=1)
    # train_data = pd.concat([train_data, end_date_1], axis=1, join="outer")

    # 删除没用的列
    data_y = pd.DataFrame(full_data["label"])
    data_x = full_data.drop(DROP_COLS, axis=1)

    # 把数据再重新按trainset和validateset切分开
    train_x = data_x.ix[:split_index - 1, :]
    validate_x = data_x.ix[split_index:, :].reset_index(drop=True)
    train_y = data_y.ix[:split_index - 1, :]
    validate_y = data_y.ix[split_index:, :].reset_index(drop=True)
    return train_x, train_y, validate_x, validate_y


def pca_method(train_x, train_y, validate_x, validate_y,  pca_threshold, is_auto=1, is_split=1):
    # 缺失值填充
    train_x = train_x.fillna(0)
    train_x = train_x.values
    validate_x = validate_x.fillna(0)
    validate_x = validate_x.values

    # 归一化，之前必须保证没有空值，之后自动变成ndarray
    # scaler = MinMaxScaler()
    # train_x = scaler.fit_transform(train_x)
    # validate_x = scaler.fit_transform(validate_x)

    # dataframe变成没有标签的ndarray，以便可以输入模型
    train_y = train_y.values
    validate_y = validate_y.values

    if is_split == 1:
        # 先把onehot列单独拿出来
        onehot_train_x_left = train_x[:, :30]
        train_x_mid = train_x[:, 30:454]
        # onehot_train_x_right = train_x[:, 454:]
        onehot_validate_x_left = validate_x[:, :30]
        validate_x_mid = validate_x[:, 30:454]
        # onehot_validate_x_right = validate_x[:, 454:]
    else:
        train_ts_code_1 = train_x[:,0]
        train_x_mid = train_x[:, 1:]
        valid_ts_code_1 = validate_x[:,0]
        validate_x_mid = validate_x[:, 1:]

    # PCA
    if is_auto == 1:
        pca = PCA(n_components='mle', whiten=False)
    else:
        pca = PCA(n_components=pca_threshold, whiten=False)
    selected_train_x = pca.fit(train_x_mid).transform(train_x_mid)
    print(pca.explained_variance_ratio_)
    selected_validate_x = pca.fit(validate_x_mid).transform(validate_x_mid)
    print(pca.explained_variance_ratio_)

    # 把ts_code再重新拼回来
    if is_split == 1:
        selected_train_x = np.hstack((onehot_train_x_left, selected_train_x))
        selected_validate_x = np.hstack((onehot_validate_x_left, selected_validate_x))
    else:
        # print(train_ts_code_1.reshape(-1,1).shape)
        # print(selected_train_x.shape)
        selected_train_x = np.hstack((train_ts_code_1.reshape(-1,1), selected_train_x))
        selected_validate_x = np.hstack((valid_ts_code_1.reshape(-1,1), selected_validate_x))

    return selected_train_x, train_y, selected_validate_x, validate_y


def factor_analysis_method(train_x, train_y, validate_x, validate_y, fa_threshold, is_split=1):
    # 缺失值填充
    train_x = train_x.fillna(0)
    train_x = train_x.values
    validate_x = validate_x.fillna(0)
    validate_x = validate_x.values

    # # 归一化，之前必须保证没有空值，之后自动变成ndarray
    # scaler = MinMaxScaler()
    # train_x = scaler.fit_transform(train_x)
    # validate_x = scaler.fit_transform(validate_x)

    # dataframe变成没有标签的ndarray，以便可以输入模型
    train_y = train_y.values
    validate_y = validate_y.values

    if is_split == 1:
        # 先把onehot列单独拿出来
        onehot_train_x_left = train_x[:, :30]
        train_x_mid = train_x[:, 30:454]
        # onehot_train_x_right = train_x[:, 454:]
        onehot_validate_x_left = validate_x[:, :30]
        validate_x_mid = validate_x[:, 30:454]
        # onehot_validate_x_right = validate_x[:, 454:]
    else:
        train_ts_code_1 = train_x[:, 0]
        train_x_mid = train_x[:, 1:]
        valid_ts_code_1 = validate_x[:, 0]
        validate_x_mid = validate_x[:, 1:]

    # factor_analysis
    fa = FactorAnalysis(n_components=fa_threshold)
    selected_train_x = fa.fit(train_x_mid).transform(train_x_mid)
    selected_validate_x = fa.fit(validate_x_mid).transform(validate_x_mid)

    # 把ts_code再重新拼回来
    if is_split == 1:#ts_code有30列
        selected_train_x = np.hstack((onehot_train_x_left, selected_train_x))
        selected_validate_x = np.hstack((onehot_validate_x_left, selected_validate_x))
    else:#ts_code只有一列
        # print(train_ts_code_1.reshape(-1,1).shape)
        # print(selected_train_x.shape)
        selected_train_x = np.hstack((train_ts_code_1.reshape(-1, 1), selected_train_x))
        selected_validate_x = np.hstack((valid_ts_code_1.reshape(-1, 1), selected_validate_x))
    return selected_train_x, train_y, selected_validate_x, validate_y


def chi_method(train_x, train_y, validate_x, validate_y,  chi_threshold, is_split=1):
    # 缺失值填充
    train_x = train_x.fillna(0)
    train_x = train_x.values
    validate_x = validate_x.fillna(0)
    validate_x = validate_x.values

    # # 归一化，之前必须保证没有空值，之后自动变成ndarray
    scaler = MinMaxScaler()
    # train_x = scaler.fit_transform(train_x)
    # validate_x = scaler.fit_transform(validate_x)

    # dataframe变成没有标签的ndarray，以便可以输入模型
    train_y = train_y.values
    validate_y = validate_y.values

    if is_split == 1:
        # 先把onehot列单独拿出来
        onehot_train_x_left = train_x[:, :30]
        train_x_mid = scaler.fit_transform(train_x[:, 30:454])
        # onehot_train_x_right = train_x[:, 454:]
        onehot_validate_x_left = validate_x[:, :30]
        validate_x_mid = scaler.fit_transform(validate_x[:, 30:454])
        # onehot_validate_x_right = validate_x[:, 454:]
        pass
    else:
        train_ts_code_1 = train_x[:, 0]
        train_x_mid = scaler.fit_transform(train_x[:, 1:])
        valid_ts_code_1 = validate_x[:, 0]
        validate_x_mid = scaler.fit_transform(validate_x[:, 1:])

    # 卡方检验法，注意，这个是针对分类问题使用的
    # 选择K个最好的特征，返回选择特征后的数据
    # temp_result = SelectKBest(chi2, k=chi_threshold).fit(train_x_mid, train_y)
    # selected_index_list = temp_result.get_support(indices=True)
    # score_list = temp_result.scores_.tolist()
    # score_dict = {}
    # for index, label in enumerate(feat_labels):
    #     print(str(index) + " : " + str(label))
    #     score_dict[label] = score_list[index]
    # sorted_score_dict = sorted(score_dict.items(), key=lambda item: item[1], reverse=True)
    selected_train_x = SelectKBest(chi2, k=chi_threshold).fit_transform(train_x_mid, train_y)
    selected_validate_x = SelectKBest(chi2, k=chi_threshold).fit_transform(validate_x_mid, validate_y)

    # 把ts_code再重新拼回来
    if is_split == 1:
        selected_train_x = np.hstack((onehot_train_x_left, selected_train_x))
        selected_validate_x = np.hstack((onehot_validate_x_left, selected_validate_x))
    else:#ts_code只有一列
        # print(train_ts_code_1.reshape(-1,1).shape)
        # print(selected_train_x.shape)
        selected_train_x = np.hstack((train_ts_code_1.reshape(-1, 1), selected_train_x))
        selected_validate_x = np.hstack((valid_ts_code_1.reshape(-1, 1), selected_validate_x))
    return selected_train_x, train_y, selected_validate_x, validate_y


def mic_method(data_x, data_y, feat_labels, mic_threshold, is_split=1):
    # 缺失值填充
    # data_x = data_x.fillna(data_x.mean())
    data_x = data_x.fillna(0)
    data_x = data_x.values
    # # 归一化，之前必须保证没有空值，之后自动变成ndarray
    # scaler = MinMaxScaler()
    # data_x = scaler.fit_transform(data_x)
    # dataframe变成没有标签的ndarray，以便可以输入模型
    data_y = data_y.values

    if is_split == 1:
        # 先把onehot列单独拿出来
        onehot_data_x_left = data_x[:, :30]
        data_x_mid = data_x[:, 30:454]
        onehot_data_x_right = data_x[:, 454:]
    else:
        data_x_mid = data_x

    # 最大信息系数法，注意，这个是针对分类问题使用的
    # 选择K个最好的特征，返回选择特征后的数据
    m = MINE()
    x = np.random.uniform(-1, 1, 10000)
    # m.compute_score(x, x ** 2)
    m.compute_score(data_x_mid, data_y)
    print(m.mic())
    xxx = SelectKBest(chi2, k=mic_threshold).fit(data_x_mid, data_y)
    selected_data_x = SelectKBest(chi2, k=mic_threshold).fit_transform(data_x_mid, data_y)
    return selected_data_x, data_y


def correlation_method(data_x, data_y, feat_labels, select_type=3, is_split=1):
    # 缺失值填充
    # data_x = data_x.fillna(data_x.mean())
    data_x = data_x.fillna(0)
    data_x = data_x.values
    # 归一化，之前必须保证没有空值，之后自动变成ndarray
    # scaler = MinMaxScaler()
    # data_x = scaler.fit_transform(data_x)
    # dataframe变成没有标签的ndarray，以便可以输入模型
    data_y = data_y.values

    if is_split == 1:
        # 先把onehot列单独拿出来
        onehot_data_x_left = data_x[:, :30]
        data_x_mid = data_x[:, 30:454]
        onehot_data_x_right = data_x[:, 454:]
    else:
        data_x_mid = data_x

    if select_type == 1:
        # 方差选择法
        # 原理：方差太低说明该列数据没有区分性，所以要删掉方差太小的特征
        # 情况：不归一化的话，方差都特大，筛不掉多少。归一化以后，方差都特小，全被筛掉了。正常数据好像是不归一化。
        selected_data_x = VarianceThreshold(0.001).fit(data_x_mid)
        vs = selected_data_x.variances_
        print(selected_data_x.variances_)
        selected_data_x = selected_data_x.transform(data_x_mid)
        print("Done")
    elif select_type == 2:
        # 相关系数法，注意，这个是针对回归问题使用的
        selected_data_x = SelectKBest(lambda X, Y: np.array(list(map(lambda x: pearsonr(x, Y), X.T))).T, k=10)\
            .fit_transform(data_x_mid, data_y)
    else:
        # 互信息法
        selected_data_x = 0

    return selected_data_x, data_y


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


# if __name__ == '__main__':
#     # 处理好x和y
#     # data_x, data_y, data = process_data()
#     train_x, train_y, validate_x, validate_y = process_data()
#     # 获取所有特征名
#     feat_labels = train_x.columns.values.tolist()
#
#     # 以下各特征选择方法函数的参数解释:
#     # data_x, data_y 是dataframe类型的数据
#     # feat_labels 是特征名列表
#     # is_auto=1 表示让算法自动决定保留的特征数
#     # is_split=1 表示对输入的data_x切割出连续型特征列，否则为不切割全部使用
#
#     # PCA，返回的数据中，数值型列没有再次进行归一化
#     pca_train_x, pca_train_y, pca_validate_x, pca_validate_y = pca_method(train_x, train_y,
#                                                                           validate_x, validate_y, feat_labels,
#                                                                           10, is_auto=0, is_split=1)
#
#     # 因子分析
#     fa_train_x, fa_train_y, fa_validate_x, fa_validate_y = factor_analysis_method(train_x, train_y,
#                                                                                   validate_x, validate_y, feat_labels,
#                                                                                   10, is_split=1)
#
#     # 卡方检验
#     chi_train_x, chi_train_y, chi_validate_x, chi_validate_y = chi_method(train_x, train_y,
#                                                                           validate_x, validate_y, feat_labels,
#                                                                           10, is_split=1)
#
#     # 以下有bug
#     # 最大信息系数
#     # data_x_mic, data_y_mic = mic_method(data_x, data_y, feat_labels, 10, is_split=1)
#     # 其他相关性方法
#     # correlation_importance_dict = correlation_method(data_x, data_y, feat_labels, select_type=1, is_split=1)
#     # 使用随机森林进行特征选择
#     # random_forest_importance_dict = random_forest_method(data_x, data_y, feat_labels)
#     # 使用遗传算法
#     # ga_importance_dict = ga_method(data_x, data_y, feat_labels)
#
#     print("Done")
