#coding:utf-8
#采用新数据，新逻辑，进行预测

#训练集采用20180331的财报行为得到20186月的涨跌
#验证集采用20180630的财报行为得到20189月的涨跌

#TARGET_PATH = "../../data/Train&Test/C1S1_oldlabel/"#y是涨跌
#TARGET_PATH = "../../data/Train&Test/C1S1_newlabel/"#改名叫C1S1了
#TARGET_PATH = "../../data/Train&Test/C2S1_newlabel/"
# TARGET_PATH = "../../data/Train&Test/C4S1_newlabel/"


import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import classification_report,roc_auc_score
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.utils import shuffle
from sklearn import preprocessing
import feature_selection as fs

#
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')

#数据处理函数
def data_process(train_file,valid_file):
    # 训练集数据处理
    f1 = open(train_file)
    traindata = pd.read_csv(f1)
    #把后半部分的行情数据去掉
    colnames=traindata.columns.values.tolist()
    labelindex=colnames.index("label")
    traindata=traindata.ix[:,:labelindex+1]

    # 计数正负样本比例
    cunt = traindata[traindata.label == 1]["ts_code"]  # 192正例，共1710
    train_zhengnum = cunt.shape[0]
    train_funum = traindata.shape[0] - train_zhengnum
    # 加以区分SH和SZ的股票
    traindata["type"] = traindata["ts_code"].map(lambda x: 'SZ' if 'SZ' in x else 'SH')
    stocktype = pd.get_dummies(traindata["type"], prefix="type")
    traindata = traindata.drop(["type"], axis=1)
    traindata = pd.concat([traindata, stocktype], axis=1, join="outer")
    # 提取x，y
    train_y = pd.DataFrame(traindata["label"])
    dropcol = ["label", "ann_date_1", "f_ann_date_1", "end_date_1"]
    train_x = traindata.drop(dropcol, axis=1)
    # 缺失值填充，数据标准化、缩放
    train_x = train_x.fillna(0)
    colnames = train_x.columns.values.tolist()
    scaler = StandardScaler()
    for colname in colnames:
        if colname == "ts_code":
            pass
        else:
            train_x[colname] = scaler.fit_transform(np.array(train_x[colname]).reshape(-1, 1))

    # 验证集数据处理
    f2 = open(valid_file)
    validdata = pd.read_csv(f2)
    # 把后半部分的行情数据去掉
    colnames = validdata.columns.values.tolist()
    labelindex = colnames.index("label")
    validdata = validdata.ix[:, :labelindex + 1]

    # 计数正负比例
    val_zheng = validdata[validdata.label == 1]
    val_fu = validdata[validdata.label == 0]
    cunt2 = validdata[validdata.label == 1]["ts_code"]  # 481正例
    # # 调整验证集的分布
    # val_zhengnum = cunt2.shape[0]
    # val_funum = validdata.shape[0] - val_zhengnum
    # val_new_zheng = int(val_funum * (train_zhengnum / train_funum))
    # select_val_zheng = val_zheng.sample(n=val_new_zheng)
    # validdata = pd.concat([select_val_zheng, val_fu], axis=0).reset_index(drop=True)
    # 加以区分SH和SZ的股票
    validdata["type"] = validdata["ts_code"].map(lambda x: 'SZ' if 'SZ' in x else 'SH')
    stocktype = pd.get_dummies(validdata["type"], prefix="type")
    validdata = validdata.drop(["type"], axis=1)
    validdata = pd.concat([validdata, stocktype], axis=1, join="outer")
    # 提取x，y
    valid_y = pd.DataFrame(validdata["label"])
    dropcol = ["label", "ann_date_1", "f_ann_date_1", "end_date_1"]
    valid_x = validdata.drop(dropcol, axis=1)
    # 缺失值填充，数据标准化、缩放
    valid_x = valid_x.fillna(0)
    colnames = valid_x.columns.values.tolist()
    scaler = StandardScaler()
    for colname in colnames:
        if colname == "ts_code":
            pass
        else:
            valid_x[colname] = scaler.fit_transform(np.array(valid_x[colname]).reshape(-1, 1))
    return train_x,train_y,valid_x,valid_y





# 结果分析函数
def result_analyze(predictions, y_validate, eval_result,targetpath,validfile,newlabel2path):
    # 取出predictions中的结果
    predictions_dict = {"class_ids": [], "logits": [], "probabilities": []}
    for pred in predictions:
        predictions_dict["class_ids"].append(pred['class_ids'][0])
        predictions_dict["logits"].append(pred['logits'][0])
        predictions_dict["probabilities"].append(pred['probabilities'][0])

    # 计算预测出1和0的数量
    class_ids = []
    for cid in predictions_dict["class_ids"]:
        class_ids.append(int(cid))
    print("预测出1的数量：", class_ids.count(1))
    print("预测出0的数量：", class_ids.count(0))

    #验证集的时间
    valid_time=int(targetpath[-9:-1])

    # 计算收益率(written by lzn)
    pre = np.array(class_ids)
    df_pre = pd.DataFrame(pre, columns=["pre_y"])
    valid_set = pd.read_csv(validfile)
    valid_set_pre = pd.concat([valid_set, df_pre], axis=1)
    select_ts_code = list(valid_set_pre[valid_set_pre["pre_y"] == 1]["ts_code"])  #####选出预测为1的股票列表
    all_change = 0  #####所有选出的股票的收益值，为分子
    s_date_2_close_sum = 0  #####所有选出股票上个月的close和，为分母
    for each_stock in select_ts_code:
        path = newlabel2path + each_stock[:-3] + "_" + each_stock[-2:] + ".csv"
        stock_df = pd.read_csv(path)
        each_stock_change = stock_df[stock_df["jidu_date"] == valid_time]["s_change"].tolist()[0]
        all_change += each_stock_change
        s_date_2_close = stock_df[stock_df["jidu_date"] == valid_time]["s_date_2_close"].tolist()[0]
        s_date_2_close_sum += s_date_2_close
    try:
        final_shouyilv = all_change / s_date_2_close_sum
    except Exception as e:
        final_shouyilv = "没有推荐出任何可购入的股票！"
    print("最终收益率：", final_shouyilv)

    # 计算混淆矩阵(written by lzn)
    confmat = classification_report(y_true=y_validate, y_pred=pre)
    print(confmat)
    print("预测结果标签：", predictions_dict["class_ids"])
    print("预测原结果：", predictions_dict["probabilities"])
    precision = eval_result["precision"]
    recall = eval_result["recall"]

    try:
        print('Test set auc:', roc_auc_score(y_validate.reshape(-1), pre))
        print('Test set f1:', 2 * precision * recall / (precision + recall))
    except Exception as e:
        print("由于没有推荐出任何可购入的股票， 所以无法计算auc和F1值！")
    return final_shouyilv,precision,recall,2 * precision * recall / (precision + recall),roc_auc_score(y_validate.reshape(-1), pre)



def main(trainfile,validfile,targetpath,newlabel2path):


    train_x,train_y,valid_x,valid_y=data_process(trainfile,validfile)

##########################################################################################
    #如果不降维，后面都不需要
    #pca降维
    train_x, train_y, valid_x, valid_y = fs.pca_method(train_x, train_y, valid_x, valid_y, pca_threshold=10, is_auto=0,is_split=0)#ts_code只有一列，没有进行哈希
    #train_x, train_y, valid_x, valid_y = fs.factor_analysis_method(train_x, train_y, valid_x, valid_y, fa_threshold=10,is_split=0)
    #train_x, train_y, valid_x, valid_y = fs.chi_method(train_x, train_y, valid_x, valid_y, chi_threshold=10, is_split=0)
    #降维后归一化
    scaler = MinMaxScaler()
    train_x[:,1:] = scaler.fit_transform(train_x[:,1:])
    valid_x[:,1:] = scaler.fit_transform(valid_x[:,1:])
    #把ndarray重新转成datafrmae
    ndcol = ["ts_code"]
    for i in range(10):
        s = "feature_pca_" + str(i)
        ndcol.append(s)

    train_x = pd.DataFrame(train_x, columns=ndcol)
    valid_x = pd.DataFrame(valid_x , columns=ndcol)
############################################################################

    # print(train_x.info())
    print(train_x.shape)#440列


    def dataframetodict(df):
        df=df.fillna(0)
        re = {}
        colnames = df.columns.values.tolist()
        for colname in colnames:
            re[colname] = np.array(df[colname])
        return re
    #train_x=dataframetodict(train_x)
    #print(train_x)


    my_feature_columns=[]
    for key in train_x.keys():
        if key=="ts_code":
            column1=tf.feature_column.categorical_column_with_hash_bucket(key="ts_code",hash_bucket_size = 30)
            column1=tf.feature_column.indicator_column(column1)
            my_feature_columns.append(column1)
        else:
            col=tf.feature_column.numeric_column(key=key)
            my_feature_columns.append(col)


    def train_input_fn(features, labels, batch_size):
        """An input function for training"""
        # Convert the inputs to a Dataset.
        features=dataframetodict(features)
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))

        # Shuffle, repeat, and batch the examples.
        dataset = dataset.shuffle(100000).repeat().batch(batch_size)
        print(dataset.output_shapes)
        # Return the dataset.
        return dataset


    def eval_input_fn(features, labels, batch_size):
        """An input function for evaluation or prediction"""
        #features=dict(features)
        features = dataframetodict(features)
        if labels is None:
            # No labels, use only features.
            inputs = features
        else:
            inputs = (features, labels)

        # Convert the inputs to a Dataset.
        dataset = tf.data.Dataset.from_tensor_slices(inputs)

        # Batch the examples
        assert batch_size is not None, "batch_size must not be None"
        dataset = dataset.batch(batch_size)

        # Return the dataset.
        return dataset

    my_checkpointing_config = tf.estimator.RunConfig(
            save_checkpoints_steps=100,  # Retain the 10 most recent checkpoints.
            keep_checkpoint_max=50,
        )
     # Build 2 hidden layer DNN with 10, 10 units respectively.
    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
            # Two hidden layers of 30 nodes each.
            hidden_units=[30],
            # The model must choose between 3 classes.
            n_classes=2,
        # optimizer=tf.train.AdamOptimizer(
        #     learning_rate=1e-7
        # ),
        #model_dir="./DNN_hash=30_batch=50_epoch=5000_shsz_tongfb",
        # config=my_checkpointing_config,
    )
            #model_dir="./model")



        # Train the Model.
    classifier.train(
            input_fn=lambda :train_input_fn(train_x,np.array(train_y),50),steps=5000)

        # Evaluate the model.
    eval_result = classifier.evaluate(
            input_fn=lambda :eval_input_fn(valid_x,np.array(valid_y),50))


    predictions=classifier.predict(input_fn=lambda :eval_input_fn(valid_x,labels=None,batch_size=50))
    predictions = list(predictions)
    # 对模型的验证和预测结果进行分析
    final_shouyilv,precision,recall,f1,auc=result_analyze(predictions, valid_y, eval_result,targetpath,validfile,newlabel2path)

    print("MODEL DONE")
    return final_shouyilv,precision,recall,f1,auc


# if __name__ == '__main__':
#     tf.logging.set_verbosity(tf.logging.INFO)
#     tf.app.run(main)


