#coding:utf-8
#使用lstm 老办法 ，placeholder那种

TARGET_PATH = "../../data/Train&Test/C1S4_newlabel_hybrid_timestep3/"#竖着拼四个
#TARGET_PATH = "../../data/Train&Test/C1S2_newlabel/"
TRAIN_FILE_NAME="hybrid_train_set.csv"
TRAIN_FILE=TARGET_PATH+TRAIN_FILE_NAME
VALID_FILE_NAME="hybrid_validate_set.csv"
VALID_FILE= TARGET_PATH+VALID_FILE_NAME
NEW_LABEL2_PATH="../../data/Common/New_Label2/"

import functools
import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score




# 处理训练集
f1 = open(TRAIN_FILE)
traindata = pd.read_csv(f1)
train_y = traindata["label"]
train_y=np.array(train_y).reshape(-1,1)
traindata = traindata.ix[:, -9:]
print(traindata.info())
# 构造[batch_size,time_len,input_size]形状的input tensor
train_x = []
scaler = StandardScaler()
for index, row in traindata.iterrows():
    colnames = traindata.columns.values.tolist()
    trax = []
    for colname in colnames:
        each = row[colname]
        each = each[1:-1].split(",")
        each = [float(x) for x in each]
        trax.append(each)
    trax = np.array(trax).T
    trax = scaler.fit_transform(trax)  # 特征归一化 统一量纲
    train_x.append(trax)
train_x = np.array(train_x).astype(np.float32)
a=train_x[0]
b=len(train_y)

    # 处理验证集
f2 = open(VALID_FILE)
validdata = pd.read_csv(f2)
valid_y = validdata["label"]
valid_y=np.array(valid_y).reshape(-1,1)
validdata = validdata.ix[:, -9:]
# 构造[batch_size,time_len,input_size]形状的input tensor
valid_x = []
scaler = StandardScaler()
for index, row in validdata.iterrows():
    colnames = validdata.columns.values.tolist()
    trax = []
    for colname in colnames:
        each = row[colname]
        each = each[1:-1].split(",")
        each = [float(x) for x in each]
        trax.append(each)
    trax = np.array(trax).T
    trax = scaler.fit_transform(trax)  # 特征归一化 统一量纲
    valid_x.append(trax)
valid_x = np.array(valid_x).astype(np.float32)


#划分batch,得到batch_index
def get_batch_index(train_x,batch_size,time_step):
    batch_index=[]
    for i in range(len(train_x)-time_step):
       if i % batch_size==0:
           batch_index.append(i)
    batch_index.append((len(train_x)-time_step))
    return batch_index



# 设置用到的参数
lr = 1e-3


# 输入数据是28维 一行 有28个像素
input_size = 9
# 时序持续时长为28  每做一次预测，需要先输入28行
timestep_size = 3
# 每个隐含层的节点数
hidden_size = 128
# LSTM的层数
layer_num = 1
# 最后输出的分类类别数量，如果是回归预测的呼声应该是1
class_num = 1

X = tf.placeholder(tf.float32, [None, timestep_size,input_size])
y = tf.placeholder(tf.float32, [None, class_num])
keep_prob = tf.placeholder(tf.float32)

# 定义一个LSTM结构， 把784个点的字符信息还原成28*28的图片

def unit_lstm():
    # 定义一层LSTM_CELL hiddensize 会自动匹配输入的X的维度
    lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, forget_bias=1.0, state_is_tuple=True)
    # 添加dropout layer， 一般只设置output_keep_prob
    lstm_cell = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
    return lstm_cell
# 调用MultiRNNCell来实现多层 LSTM
mlstm_cell = tf.nn.rnn_cell.MultiRNNCell([unit_lstm() for i in range(layer_num)], state_is_tuple=True)
batch_size=tf.shape(X)[0]
# 使用全零来初始化state
init_state = mlstm_cell.zero_state(batch_size, dtype=tf.float32)
outputs, state = tf.nn.dynamic_rnn(mlstm_cell, inputs=X, initial_state=init_state,
                                   time_major=False)
######outputs: The RNN output Tensor.If time_major == False (default), this will be a Tensor shaped: [batch_size, max_time, cell.output_size]
h_state = outputs[:, -1, :]
# outputs = state.h
#
# logits = tf.layers.dense(inputs=outputs, units=1)

# 设置loss function 和优化器
W = tf.Variable(tf.truncated_normal([hidden_size, class_num], stddev=0.1), dtype=tf.float32)
bias = tf.Variable(tf.random_normal(shape=[class_num]), dtype=tf.float32)
y_pre = tf.nn.sigmoid(tf.matmul(h_state, W) + bias)
# class_ids=[0]*len(list(y_pre))
# for i in range(list(y_pre)):
#     if list(y_pre)[i]>0.5:
#         class_ids[i]=1
#     else:
#         class_ids[i]=0
# 损失和评估函数
cross_entropy = -tf.reduce_mean(y * tf.log(y_pre))
train_op = tf.train.AdamOptimizer(lr).minimize(cross_entropy)



print(train_x.shape)
print(train_y.shape)
# 开始训练
sess = tf.Session()
sess.run(tf.global_variables_initializer())
batch_index=get_batch_index(train_x,batch_size=50,time_step=timestep_size)
for i in range(10):
    for step in range(len(batch_index) - 1):
        _, loss_ = sess.run([train_op, cross_entropy], feed_dict={X: train_x[batch_index[step]:batch_index[step + 1]],
                                                         y: train_y[batch_index[step]:batch_index[step + 1]],keep_prob: 0.5})
    print(i, loss_)
    # if (i+1)%200 == 0:
    # train_accuracy  = sess.run(accuracy, feed_dict={X: train_x[batch_index[step]:batch_index[step + 1]],
    #                                                      y: train_y[batch_index[step]:batch_index[step + 1]],keep_prob: 0.5})
    # print("step %d, training accuracy %g" % ((i+1), train_accuracy ))
    #     sess.run(train_op, feed_dict={_X: batch[0], y: batch[1], keep_prob: 0.5,
    #                                   batch_size: _batch_size})
# pre=sess.run(class_ids,feed_dict={X: valid_x, y: valid_y, keep_prob: 1.0})
# h_state=sess.run(h_state,feed_dict={X: valid_x, y: valid_y, keep_prob: 1.0})
# print(h_state)
pre=sess.run(y_pre,feed_dict={X: valid_x, y: valid_y, keep_prob: 1.0})
print("预测结果:",pre)
pre=list(pre)
for i in range(len(pre)):
    if pre[i]>0.5:
        pre[i]=1
    else:
        pre[i]=0
# print("yuce:",pre2)

print("预测出1的数量：", list(pre).count(1))
pre = np.array(pre)


#############################################################################################
confmat = classification_report(y_true=valid_y, y_pred=pre)
print(confmat)


precision = precision_score(valid_y,pre)
recall = recall_score(valid_y,pre)
acc=accuracy_score(valid_y,pre)
print("acc:",acc)
print('Test set f1:', f1_score(valid_y,pre))
print("precision:",precision)
################################################
    #计算回报率
df_pre = pd.DataFrame(pre, columns=["pre_y"])
valid_set = pd.read_csv(VALID_FILE)
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
final_shouyilv = all_change / s_date_2_close_sum
print("最终回报率：", final_shouyilv)

# print("test accuracy %g" % sess.run(,feed_dict={_X: images, y: labels, keep_prob: 1.0,
#                                                         batch_size: mnist.test.images.shape[0]}))


