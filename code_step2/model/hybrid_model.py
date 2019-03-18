# TODO:在数据处理部分加入数据预处理的接口，比如降维
# TODO:把模型写成可以动态加入各个部件的形式
# TODO:使用tensorboard以及matplotlib进行可视化

import os
import string
import tempfile
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

TIME_STEP = 3
TARGET_PATH = "../../data/Train&Test/C1S4_newlabel_hybrid_timestep3/"
TRAIN_FILE_NAME = "hybrid_train_set.csv"
VALIDATE_FILE_NAME = "hybrid_validate_set.csv"
NEW_LABEL2_PATH = "../../data/Common/New_Label2/"
COLUMNS_TO_DROP = ["ts_code", "ann_date_1", "f_ann_date_1", "end_date_1", "report_type_1", "label", "close_1",
                   "trade_date"]
SPLIT_FEATURE_NAME = "open"


# Create a LocalCLIDebugHook and use it as a monitor when calling fit().
HOOKS = [tf_debug.LocalCLIDebugHook(ui_type="readline")]


# 数据预处理函数
def data_process(raw_train_data, raw_validate_data):
    # 取出y_train和y_validate
    y_train = np.array(raw_train_data.loc[:, "label"].tolist())
    y_validate = np.array(raw_validate_data.loc[:, "label"].tolist())

    # 取出需要的列作为x_train和y_validate
    x_train = raw_train_data.drop(COLUMNS_TO_DROP, axis=1)
    x_validate = raw_validate_data.drop(COLUMNS_TO_DROP, axis=1)

    # 对x_train和y_validate填充缺失值
    x_train = x_train.fillna(0)
    x_validate = x_validate.fillna(0)

    # 对x_train和y_validate进行分割，left是非时序特征，right是时序特征
    split_point = x_train.columns.tolist().index(SPLIT_FEATURE_NAME)
    x_train_left = x_train.ix[:, :split_point]
    x_train_right = x_train.ix[:, split_point:]
    x_validate_left = x_validate.ix[:, :split_point]
    x_validate_right = x_validate.ix[:, split_point:]

    # 对x_train和y_validate的左半部分进行标准化和归一化
    std_scaler = StandardScaler()
    minmax_scaler = MinMaxScaler()
    col_names_left = x_train_left.columns.values.tolist()
    for col_name in col_names_left:
        x_train_left[col_name] = std_scaler.fit_transform(np.array(x_train_left[col_name]).reshape(-1, 1))
        x_validate_left[col_name] = std_scaler.fit_transform(np.array(x_validate_left[col_name]).reshape(-1, 1))
        x_train_left[col_name] = minmax_scaler.fit_transform(np.array(x_train_left[col_name]).reshape(-1, 1))
        x_validate_left[col_name] = minmax_scaler.fit_transform(np.array(x_validate_left[col_name]).reshape(-1, 1))

    # 对x_train和y_validate的右半部分进行标准化和归一化
    # 先重构x_train成为x_train_right_ndarray，重构x_validate成为x_validate_right_ndarray
    # [batch_size x sentence_size x embedding_size]
    x_train_right_ndarray = np.ndarray(shape=(x_train_right.shape[0], TIME_STEP, x_train_right.shape[1]))
    x_validate_right_ndarray = np.ndarray(shape=(x_validate_right.shape[0], TIME_STEP, x_validate_right.shape[1]))
    for index_x in range(x_train_right.shape[0]):
        one_line = x_train_right.ix[index_x, :]
        for index_z in range(x_train_right.shape[1]):
            one_feature_str = one_line[index_z]
            one_feature_list = one_feature_str.replace("[", "").replace("]", "").replace(" ", "").split(",")
            for index_y in range(TIME_STEP):
                x_train_right_ndarray[index_x][index_y][index_z] = float(one_feature_list[index_y])
    for index_x in range(x_validate_right.shape[0]):
        one_line = x_validate_right.ix[index_x, :]
        for index_z in range(x_validate_right.shape[1]):
            one_feature_str = one_line[index_z]
            one_feature_list = one_feature_str.replace("[", "").replace("]", "").replace(" ", "").split(",")
            for index_y in range(TIME_STEP):
                x_validate_right_ndarray[index_x][index_y][index_z] = float(one_feature_list[index_y])
    # 然后对x_train_right_ndarray和x_validate_right_ndarray进行标准化
    for feature_index in range(x_train_right.shape[1]):
        for step_index in range(TIME_STEP):
            raw_ndarray = \
                x_train_right_ndarray[:, step_index:step_index + 1, feature_index:feature_index + 1].reshape(-1, 1)
            std_ndarray = std_scaler.fit_transform(raw_ndarray)
            minmax_ndarray = minmax_scaler.fit_transform(std_ndarray)
            ok_ndarray = minmax_ndarray.reshape(-1, 1, 1)
            x_train_right_ndarray[:, step_index:step_index + 1, feature_index:feature_index + 1] = ok_ndarray
    for feature_index in range(x_validate_right.shape[1]):
        for step_index in range(TIME_STEP):
            raw_ndarray = \
                x_validate_right_ndarray[:, step_index:step_index + 1, feature_index:feature_index + 1].reshape(-1, 1)
            std_ndarray = std_scaler.fit_transform(raw_ndarray)
            minmax_ndarray = minmax_scaler.fit_transform(std_ndarray)
            ok_ndarray = minmax_ndarray.reshape(-1, 1, 1)
            x_validate_right_ndarray[:, step_index:step_index + 1, feature_index:feature_index + 1] = ok_ndarray

    # 降维，特征选择等

    # dataframe转ndarray, 数据变形
    x_train_left = x_train_left.values
    x_validate_left = x_validate_left.values
    y_train = y_train.reshape(-1, 1)
    y_validate = y_validate.reshape(-1, 1)

    return x_train_left, x_train_right_ndarray, y_train, x_validate_left, x_validate_right_ndarray, y_validate


# estimator执行任务函数
def estimator_op(logits, labels, mode):
    # 在Estimator传入tf.estimator.ModeKeys.PREDICT时，函数将自动传入labels = None
    loss = None
    if labels is not None:
        # 把存有labels的ndarray给reshape成(?, 1)的形状，以和模型输出保持一致
        labels = tf.reshape(labels, [-1, 1])

        # 仅在train和validate时使用交叉熵损失函数计算个batch的平均损失，因为predict时的labels自动为None
        # 计算loss时，输入的logits是经过激活函数之前的模型输出值
        loss = tf.losses.sigmoid_cross_entropy(labels, logits)

    # 得到预测y值列表
    # 先将logits通过sigmoid转换到0和1之间
    # tf.round()是四舍五入函数，适合二分类
    # 如果是shape=(batch, n)的，即n分类问题，则使用tf.argmax(probabilities, axis=1, name='predict')
    # 注意，logits的shape=(?, 1)，是一个二维tensor，但是我们需要得到的predicted_classes需要是一个shape=(?,)的一维tensor
    probabilities = tf.nn.sigmoid(logits)
    probabilities_re = tf.reshape(probabilities, [-1])
    predicted_classes = tf.round(probabilities_re, name='predict')
    predicted_classes = tf.cast(predicted_classes, tf.int32, name="cast")

    # 对应三个过程的处理
    if mode == tf.estimator.ModeKeys.TRAIN:
        # 创建优化器
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        # 构建训练操作
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
    elif mode == tf.estimator.ModeKeys.EVAL:
        # 计算准确率(accuracy)
        accuracy = tf.metrics.accuracy(labels=labels,
                                       predictions=predicted_classes,
                                       name='acc_op')
        precision = tf.metrics.precision(labels=labels,
                                         predictions=predicted_classes,
                                         name='acc_op')
        recall = tf.metrics.recall(labels=labels, predictions=predicted_classes, name='acc_op')
        metrics = {'accuracy': accuracy, 'precision': precision, 'recall': recall}
        # 使对应的内容能在tensorboard中绘制出来
        tf.summary.scalar('accuracy', accuracy[1])
        tf.summary.scalar('precision', precision[1])
        tf.summary.scalar('recall', recall[1])
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)
    else:
        predictions = {
            # tf.newaxis会新建一列，如下predicted_classes的shape变成了(?, 1)
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.sigmoid(logits),
            'logits': logits
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)


# 结果分析函数
def result_analyze(predictions, y_validate, eval_result):
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

    # 计算收益率(written by lzn)
    pre = np.array(class_ids)
    df_pre = pd.DataFrame(pre, columns=["pre_y"])
    valid_set = pd.read_csv(TARGET_PATH + VALIDATE_FILE_NAME)
    valid_set_pre = pd.concat([valid_set, df_pre], axis=1)
    select_ts_code = list(valid_set_pre[valid_set_pre["pre_y"] == 1]["ts_code"])  #####选出预测为1的股票列表
    all_change = 0  #####所有选出的股票的收益值，为分子
    s_date_2_close_sum = 0  #####所有选出股票上个月的close和，为分母
    for each_stock in select_ts_code:
        path = NEW_LABEL2_PATH + each_stock[:-3] + "_" + each_stock[-2:] + ".csv"
        stock_df = pd.read_csv(path)
        each_stock_change = stock_df[stock_df["jidu_date"] == 20180930]["s_change"].tolist()[0]
        all_change += each_stock_change
        s_date_2_close = stock_df[stock_df["jidu_date"] == 20180930]["s_date_2_close"].tolist()[0]
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


def main(argv):
    ####################################################数据部分########################################################

    # 解析参数
    args = arg_parser.parse_args(argv[1:])

    # 读取数据
    raw_train_data = pd.read_csv(TARGET_PATH + TRAIN_FILE_NAME)
    raw_validate_data = pd.read_csv(TARGET_PATH + VALIDATE_FILE_NAME)
    x_train_left, x_train_right_ndarray, y_train, x_validate_left, x_validate_right_ndarray, y_validate = \
        data_process(raw_train_data, raw_validate_data)

    # 创建train_input_fn()和eval_input_fn()
    def map_parser(x_left, x_right_ndarray, y):
        # 把整个380列数据起名叫做"x_left", 后面9列叫做"x_right",并且把"x_left"和"x_right"都放到features字典里
        # 注意，tf自带的column类不能接收多列特征，只能接收类似于"x_test_1"的单列特征。
        features = {"x_test_1": x_left[:, :1], "x_test_2": x_left[:, 1:2], "x_left": x_left, "x_right": x_right_ndarray}
        return features, y

    def train_input_fn():
        # (6130, 380) (6130, 3, 9) (6130, 1)拼接在一起
        dataset = tf.data.Dataset.from_tensor_slices((x_train_left, x_train_right_ndarray, y_train))
        dataset = dataset.shuffle(6500)
        # 380列，batch100，所以每次送入模型的tensor.shape是(100，380)，实际显示是(?, 380)
        dataset = dataset.batch(50)
        dataset = dataset.map(map_parser)
        dataset = dataset.repeat()
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()
        # return dataset

    def eval_input_fn():
        # (6130, 380) (6130, 3, 9) (6130, 1)拼接在一起
        dataset = tf.data.Dataset.from_tensor_slices((x_validate_left, x_validate_right_ndarray, y_validate))
        dataset = dataset.batch(50)
        dataset = dataset.map(map_parser)
        # 模型验证集合不可以进行repeat()，shuffle(6500)实际上也没有意义
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()

    ####################################################模型部分########################################################

    # 创建模型左半部分-自带的dnn
    # # 定义特征列
    # columns = [tf.feature_column.numeric_column("x_test_1"), tf.feature_column.numeric_column("x_test_2")]
    # # 模型
    # classifier = tf.estimator.DNNClassifier(
    #     hidden_units=[100],
    #     feature_columns=columns,
    #     model_dir=os.path.join("./test_model", 'bow_embeddings'))
    # classifier.train(input_fn=train_input_fn, steps=1000)
    # eval_result = classifier.evaluate(input_fn=eval_input_fn)
    # predictions = classifier.predict(input_fn=eval_input_fn)
    # # 对模型的验证和预测结果进行分析
    # result_analyze(predictions, y_validate, eval_result)
    # print("DNN OK!")

    # 创建模型左半部分-自定义dnn
    def dnn_model_fn(features, labels, mode):
        # 构造输入层，输出shape=(?, 3, 9)
        input_layer = features["x_left"]

        # dropout层，在每层后面都要加一个，输出shape=(?, 3, 9)
        # dropout_emb = tf.layers.dropout(inputs=input_layer, rate=0.2)
        dropout_emb = input_layer

        # MLP层，输出shape=(?, units)
        # units参数代表节点数
        hidden_1 = tf.layers.dense(inputs=dropout_emb, units=100, activation=tf.nn.relu)

        # dropout层，在每层后面都要加一个，输出shape=(?, 3, 9)
        # dropout_hidden_1 = tf.layers.dropout(inputs=hidden_1, rate=0.2)

        # 输出层，一个节点，输出shape=(?, 1)，值是预测出的y值，二分类的话，范围是0到1，下面根据该值和0/1的距离来确定0/1
        logits = tf.layers.dense(inputs=hidden_1, units=1)

        # 在Estimator传入tf.estimator.ModeKeys.PREDICT时，函数将自动传入labels = None
        loss = None
        if labels is not None:
            # 把存有labels的ndarray给reshape成(?, 1)的形状，以和模型输出保持一致
            labels = tf.reshape(labels, [-1, 1])

            # 仅在train和validate时使用交叉熵损失函数计算个batch的平均损失，因为predict时的labels自动为None
            # 计算loss时，输入的logits是经过激活函数之前的模型输出值
            loss = tf.losses.sigmoid_cross_entropy(labels, logits)

        # 得到预测y值列表
        predicted_classes = tf.argmax(logits, axis=1, name='predict')

        # 对应三个过程的处理
        if mode == tf.estimator.ModeKeys.TRAIN:
            # 创建优化器
            optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
            # 构建训练操作
            train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
        elif mode == tf.estimator.ModeKeys.EVAL:
            # 计算准确率(accuracy)
            accuracy = tf.metrics.accuracy(labels=labels,
                                           predictions=predicted_classes,
                                           name='acc_op')
            precision = tf.metrics.precision(labels=labels,
                                             predictions=predicted_classes,
                                             name='acc_op')
            recall = tf.metrics.recall(labels=labels, predictions=predicted_classes, name='acc_op')
            metrics = {'accuracy': accuracy, 'precision': precision, 'recall': recall}
            # 使对应的内容能在tensorboard中绘制出来
            tf.summary.scalar('accuracy', accuracy[1])
            tf.summary.scalar('precision', precision[1])
            tf.summary.scalar('recall', recall[1])
            return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)
        else:
            predictions = {
                # tf.newaxis会新建一列，如下predicted_classes的shape变成了(?, 1)
                'class_ids': predicted_classes[:, tf.newaxis],
                'probabilities': tf.nn.sigmoid(logits),
                'logits': logits
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # 创建模型右半部分-cnn
    def cnn_model_fn(features, labels, mode, params):
        # 所有层的shape中的"?"均为batch_size

        # 构造输入层，输出shape=(?, 3, 9)
        input_layer = features["x_right"]

        # dropout层，在每层后面都要加一个，输出shape=(?, 3, 9)
        dropout_emb = tf.layers.dropout(inputs=input_layer, rate=0.2)

        # 卷积层，当stride=1时，输出shape=(?, time_step + 1, filters)
        # conv = tf.layers.conv1d(
        #     inputs=dropout_emb,
        #     filters=32,
        #     kernel_size=20,
        #     padding="same",
        #     activation=tf.nn.relu)
        # filter表示生成几个feature vector，所有feature vector组成一个feature map。
        # kernel_size表示卷积核的shape=(feature_dim, kernel_size)
        # padding为"valid"时，kernel向前移动时，只要数据的最后一列已经被扫描了，卷积核就不会再移动了。
        conv = tf.keras.layers.Conv1D(filters=4, kernel_size=2, padding="valid", activation=tf.nn.relu).apply(dropout_emb)
        conv_2 = tf.keras.layers.Conv1D(filters=4, kernel_size=3, padding="valid", activation=tf.nn.relu).apply(dropout_emb)

        # Global Max Pooling层，输出shape=(?, filters)
        # 函数带reduce的说明和pooling有关，类似于"降维"的含义。max说明是选出最大值。
        pool_1 = tf.reduce_max(input_tensor=conv, axis=1)
        pool_2 = tf.reduce_max(input_tensor=conv_2, axis=1)

        # 拼接上层输出的两个tensor, 第二个参数是拼接的维度index，比如shape=(?, 2, 8)里面，2对应的维度index是1
        pool = tf.concat([pool_1, pool_2], 1)

        # MLP层，输出shape=(?, units)
        # units参数代表节点数
        hidden = tf.layers.dense(inputs=pool, units=250, activation=tf.nn.relu)

        # dropout层，在每层后面都要加一个，输出shape=(?, filters)
        dropout_hidden = tf.layers.dropout(inputs=hidden, rate=0.2)

        # 输出层，一个节点，输出shape=(?, 1)，值是预测出的y值，二分类的话，范围是0到1，下面根据该值和0/1的距离来确定0/1
        logits = tf.layers.dense(inputs=dropout_hidden, units=1)

        # 在Estimator传入tf.estimator.ModeKeys.PREDICT时，函数将自动传入labels = None
        loss = None
        if labels is not None:
            # 把存有labels的ndarray给reshape成(?, 1)的形状，以和模型输出保持一致
            labels = tf.reshape(labels, [-1, 1])

            # 仅在train和validate时使用交叉熵损失函数计算个batch的平均损失，因为predict时的labels自动为None
            # 计算loss时，输入的logits是经过激活函数之前的模型输出值
            loss = tf.losses.sigmoid_cross_entropy(labels, logits)

        # 得到预测y值列表
        predicted_classes = tf.argmax(logits, axis=1, name='predict')

        # 对应三个过程的处理
        if mode == tf.estimator.ModeKeys.TRAIN:
            # 创建优化器
            optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
            # 构建训练操作
            train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
        elif mode == tf.estimator.ModeKeys.EVAL:
            # 计算准确率(accuracy)
            accuracy = tf.metrics.accuracy(labels=labels,
                                           predictions=predicted_classes,
                                           name='acc_op')
            precision = tf.metrics.precision(labels=labels,
                                             predictions=predicted_classes,
                                             name='acc_op')
            recall = tf.metrics.recall(labels=labels, predictions=predicted_classes, name='acc_op')
            metrics = {'accuracy': accuracy, 'precision': precision, 'recall': recall}
            # 使对应的内容能在tensorboard中绘制出来
            tf.summary.scalar('accuracy', accuracy[1])
            tf.summary.scalar('precision', precision[1])
            tf.summary.scalar('recall', recall[1])
            return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)
        else:
            predictions = {
                # tf.newaxis会新建一列，如下predicted_classes的shape变成了(?, 1)
                'class_ids': predicted_classes[:, tf.newaxis],
                'probabilities': tf.nn.sigmoid(logits),
                'logits': logits
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # 创建模型右半部分-lstm
    head = tf.contrib.estimator.binary_classification_head()

    def lstm_model_fn(features, labels, mode):
        inputs = features["x_right"]

        # create an LSTM cell of size 100
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(100)

        # create the complete LSTM
        _, final_states = tf.nn.dynamic_rnn(lstm_cell, inputs, dtype=tf.float64)

        # get the final hidden states of dimensionality [batch_size x sentence_size]
        outputs = final_states.h

        logits = tf.layers.dense(inputs=outputs, units=1)

        # This will be None when predicting
        if labels is not None:
            labels = tf.reshape(labels, [-1, 1])

        # if mode == tf.estimator.ModeKeys.PREDICT:

        optimizer = tf.train.AdamOptimizer()

        def _train_op_fn(loss):
            return optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())

        return head.create_estimator_spec(
            features=features,
            labels=labels,
            mode=mode,
            logits=logits,
            train_op_fn=_train_op_fn)

    # 创建完整模型-自定义hybrid
    def hybrid_model_fn(features, labels, mode):
        # 所有层的shape中的"?"均为batch_size

        # 构造输入层-左，输出shape=(?, 380); 构造输入层-左，输出shape=(?, 3, 9)
        input_layer_left = features["x_left"]
        input_layer_right = features["x_right"]

        ####################################################################################
        # 模型的左半部分，LR

        # MLP层，输出shape=(?, units)
        # units参数代表节点数
        hidden_left_1 = tf.layers.dense(inputs=input_layer_left, units=100, activation=tf.nn.relu)

        # dropout层，在每层后面都要加一个，输出shape=(?, 3, 9)
        dropout_hidden_1eft_1 = tf.layers.dropout(inputs=hidden_left_1, rate=0.2)

        # 模型左侧的输出向量层，输出shape=(?, units)
        output_vector_left = tf.layers.dense(inputs=dropout_hidden_1eft_1, units=5, activation=tf.nn.relu)

        ####################################################################################
        # 模型的右半部分，LSTM

        # create an LSTM cell of size 100
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(100)

        # create the complete LSTM
        _, final_states = tf.nn.dynamic_rnn(lstm_cell, input_layer_right, dtype=tf.float64)

        # get the final hidden states of dimensionality [batch_size x sentence_size]，即shape=(?, 3)
        outputs = final_states.h

        # 模型右侧的输出向量层，输出shape=(?, units)
        output_vector_right = tf.layers.dense(inputs=outputs, units=5, activation=tf.nn.relu)

        ####################################################################################
        # 拼接模型的左右两部分并继续前向传播

        # 拼接上层输出的两个tensor, 第二个参数是拼接的维度index，比如shape=(?, 2, 8)里面，2对应的维度index是1
        output_vector_hybrid = tf.concat([output_vector_left, output_vector_right], 1)

        # MLP层，输出shape=(?, units)
        # units参数代表节点数
        hidden_hybrid = tf.layers.dense(inputs=output_vector_hybrid, units=10, activation=tf.nn.relu)

        # dropout层，在每层后面都要加一个，输出shape=(?, filters)
        dropout_hidden_hybrid = tf.layers.dropout(inputs=hidden_hybrid, rate=0.2)

        # 输出层，一个节点，输出shape=(?, 1)，值是预测出的y值，二分类的话，范围是0到1，下面根据该值和0/1的距离来确定0/1
        logits = tf.layers.dense(inputs=dropout_hidden_hybrid, units=1)

        # 针对不同的mode，estimator执行不同的计算，并返回结果
        return estimator_op(logits, labels, mode)

    ################################################训练、验证和分析部分################################################

    # 创建DNN模型的对象，并对模型进行训练、验证和预测
    # dnn_classifier = tf.estimator.Estimator(model_fn=dnn_model_fn, model_dir=os.path.join("./test_model", 'dnn'))
    # dnn_classifier.train(input_fn=train_input_fn, steps=1000)
    # eval_result = dnn_classifier.evaluate(input_fn=eval_input_fn)
    # predictions = dnn_classifier.predict(input_fn=eval_input_fn)

    # 创建CNN模型的对象，并对模型进行训练、验证和预测
    # cnn_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir=os.path.join("./test_model", 'cnn'))
    # cnn_classifier.train(input_fn=train_input_fn, steps=1000)
    # eval_result = cnn_classifier.evaluate(input_fn=eval_input_fn)
    # predictions = cnn_classifier.predict(input_fn=eval_input_fn)

    # 创建LSTM模型的对象，并对模型进行训练、验证和预测
    # lstm_classifier = tf.estimator.Estimator(model_fn=lstm_model_fn, model_dir=os.path.join("./test_model", 'lstm'))
    # lstm_classifier.train(input_fn=train_input_fn, steps=1000)
    # eval_result = lstm_classifier.evaluate(input_fn=eval_input_fn)
    # predictions = lstm_classifier.predict(input_fn=eval_input_fn)

    # 创建hybrid模型的对象，并对模型进行训练、验证和预测
    hybrid_classifier = tf.estimator.Estimator(model_fn=hybrid_model_fn, model_dir=os.path.join("./test_model", 'mix'))
    # hybrid_classifier.train(input_fn=train_input_fn, steps=3000)
    eval_result = hybrid_classifier.evaluate(input_fn=eval_input_fn)
    predictions = hybrid_classifier.predict(input_fn=eval_input_fn)

    # 对模型的验证和预测结果进行分析
    result_analyze(predictions, y_validate, eval_result)

    print("MODEL DONE")


if __name__ == '__main__':
    # 设置运行时log
    tf.logging.set_verbosity(tf.logging.INFO)

    # 确定输入参数
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--batch_size', default=100, type=int, help='batch size')
    arg_parser.add_argument('--train_steps', default=1000, type=int, help='number of training steps')

    # 执行main
    tf.app.run(main)
