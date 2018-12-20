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


def pca_method(data_x, data_y, feat_labels, pca_threshold, is_auto=1, is_split=1):
    # 缺失值填充
    # data_x = data_x.fillna(data_x.mean())
    data_x = data_x.fillna(0)
    data_x = data_x.values
    # 归一化，之前必须保证没有空值，之后自动变成ndarray
    scaler = MinMaxScaler()
    data_x = scaler.fit_transform(data_x)
    # dataframe变成没有标签的ndarray，以便可以输入模型
    data_y = data_y.values

    if is_split == 1:
        # 先把onehot列单独拿出来
        # onehot_data_x_left = data_x[:, :30]
        data_x_mid = data_x[:, 30:454]
        # onehot_data_x_right = data_x[:, 454:]
    else:
        data_x_mid = data_x

    # PCA
    if is_auto == 1:
        pca = PCA(n_components='mle', whiten=False)
    else:
        pca = PCA(n_components=pca_threshold, whiten=False)
    pca_data_x = pca.fit(data_x_mid).transform(data_x_mid)
    print(pca.explained_variance_ratio_)

    # 拼接成原数据集
    # data_x = np.hstack((onehot_data_x_left, pca_data_x))
    # data_x = np.hstack((data_x, onehot_data_x_right))

    return pca_data_x, data_y


def factor_analysis_method(data_x, data_y, feat_labels, fa_threshold, is_split=1):
    # 缺失值填充
    # data_x = data_x.fillna(data_x.mean())
    data_x = data_x.fillna(0)
    data_x = data_x.values
    # 归一化，之前必须保证没有空值，之后自动变成ndarray
    scaler = MinMaxScaler()
    data_x = scaler.fit_transform(data_x)
    # dataframe变成没有标签的ndarray，以便可以输入模型
    data_y = data_y.values

    if is_split == 1:
        # 先把onehot列单独拿出来
        # onehot_data_x_left = data_x[:, :30]
        data_x_mid = data_x[:, 30:454]
        # onehot_data_x_right = data_x[:, 454:]
    else:
        data_x_mid = data_x

    # factor_analysis
    fa = FactorAnalysis(n_components=fa_threshold)
    fa_data_x = fa.fit(data_x_mid).transform(data_x_mid)
    return fa_data_x, data_y


def chi_method(data_x, data_y, feat_labels, chi_threshold, is_split=1):
    # 缺失值填充
    # data_x = data_x.fillna(data_x.mean())
    data_x = data_x.fillna(0)
    data_x = data_x.values
    # 归一化，之前必须保证没有空值，之后自动变成ndarray
    scaler = MinMaxScaler()
    data_x = scaler.fit_transform(data_x)
    # dataframe变成没有标签的ndarray，以便可以输入模型
    data_y = data_y.values

    if is_split == 1:
        # 先把onehot列单独拿出来
        # onehot_data_x_left = data_x[:, :30]
        data_x_mid = data_x[:, 30:454]
        feat_labels = feat_labels[30:454]
        # onehot_data_x_right = data_x[:, 454:]
    else:
        data_x_mid = data_x

    # 卡方检验法，注意，这个是针对分类问题使用的
    # 选择K个最好的特征，返回选择特征后的数据
    temp_result = SelectKBest(chi2, k=chi_threshold).fit(data_x_mid, data_y)
    selected_index_list = temp_result.get_support(indices=True)
    score_list = temp_result.scores_.tolist()
    score_dict = {}
    for index, label in enumerate(feat_labels):
        print(str(index) + " : " + str(label))
        score_dict[label] = score_list[index]
    sorted_score_dict = sorted(score_dict.items(), key=lambda item: item[1], reverse=True)
    selected_data_x = SelectKBest(chi2, k=chi_threshold).fit_transform(data_x_mid, data_y)

    # threshold=10时，选到的特征为[5,16,80,186,292,320,321,332,334,398]
    # surplus_rese_ps_1
    # assert_turn_1
    # fixed_assets_1
    # fixed_assets_2
    # fixed_assets_3
    # total_revenue_ps_4
    # revenue_ps_4
    # ca_turn_4
    # assets_turn_4
    # fixed_assets_4

    # 排序后的特征为
    # 'ca_turn_4' :             10.706118228843835      流动资产周转率_4
    # 'assets_turn_4' :         7.817477658077262       总资产周转率_4
    # 'total_revenue_ps_4' :    5.123831282832661       每股营业总收入_4
    # 'revenue_ps_4' :          5.096661116892932       每股营业收入_4
    # 'assets_turn_1' :         3.3288899118747106      总资产周转率_1
    # 'fixed_assets_2' :        3.285283662717915       固定资产合计_2
    # 'fixed_assets_4' :        3.273297318316537       固定资产合计_4
    # 'fixed_assets_3' :        3.0832010142106023      固定资产合计_3
    # 'fixed_assets_1' :        3.024284732108817       固定资产合计_1
    # 'surplus_rese_ps_1' :     2.9725377340171093      每股盈余公积_1

    return selected_data_x, data_y


def mic_method(data_x, data_y, feat_labels, mic_threshold, is_split=1):
    # 缺失值填充
    # data_x = data_x.fillna(data_x.mean())
    data_x = data_x.fillna(0)
    data_x = data_x.values
    # 归一化，之前必须保证没有空值，之后自动变成ndarray
    scaler = MinMaxScaler()
    data_x = scaler.fit_transform(data_x)
    # dataframe变成没有标签的ndarray，以便可以输入模型
    data_y = data_y.values

    if is_split == 1:
        # 先把onehot列单独拿出来
        # onehot_data_x_left = data_x[:, :30]
        data_x_mid = data_x[:, 30:454]
        # onehot_data_x_right = data_x[:, 454:]
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


if __name__ == '__main__':
    # 处理好x和y
    data_x, data_y, data = process_data()
    # 获取所有特征名
    feat_labels = data_x.columns.values.tolist()

    # 以下各特征选择方法函数的参数解释:
    # data_x, data_y 是dataframe类型的数据
    # feat_labels 是特征名列表
    # is_auto=1 表示让算法自动决定保留的特征数
    # is_split=1 表示对输入的data_x切割出连续型特征列，否则为不切割全部使用

    # PCA，返回的数据中，数值型列没有再次进行归一化
    # data_x_pca, data_y_pca = pca_method(data_x, data_y, feat_labels, 10, is_auto=0, is_split=1)

    # 卡方检验
    data_x_chi, data_y_chi = chi_method(data_x, data_y, feat_labels, 10, is_split=1)

    # 因子分析
    # data_x_fa, data_y_fa = factor_analysis_method(data_x, data_y, feat_labels, 10, is_split=1)

    # 以下有bug
    # 最大信息系数
    # data_x_mic, data_y_mic = mic_method(data_x, data_y, feat_labels, 10, is_split=1)
    # 其他相关性方法
    # correlation_importance_dict = correlation_method(data_x, data_y, feat_labels, select_type=1, is_split=1)
    # 使用随机森林进行特征选择
    # random_forest_importance_dict = random_forest_method(data_x, data_y, feat_labels)
    # 使用遗传算法
    # ga_importance_dict = ga_method(data_x, data_y, feat_labels)

    print("Done")
