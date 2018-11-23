import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split,cross_val_score

#
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')


def main(argv):
    args = parser.parse_args(argv[1:])

    f1=open("../../data/Train&Test/full_train_set.csv")
    data=pd.read_csv(f1)
    datay=pd.DataFrame(data["label"])
    dropcol=["label","ann_date_1","ann_date_2","end_date_2","ann_date_3","end_date_3","ann_date_4","end_date_4"]
    datax=data.drop(dropcol,axis=1)
    end_date_1=pd.get_dummies(datax["end_date_1"],prefix="end_date")
    datax=datax.drop(["end_date_1"],axis=1)
    datax=pd.concat([datax,end_date_1],axis=1,join="outer")
    print(datax.shape)#440列
    f1.close()

    #读取训练标签，赋予列明

    #



    # 划分训练集，验证集
    train_x, valid_x, train_y, valid_y = train_test_split(datax, datay, test_size=0.25, random_state=100)  # 默认0.25的验证集

    # train_x=datax[:15000]
    # train_y=datay[:15000]
    # valid_x=datax[15000:]
    # valid_y=datay[15000:]
    def dataframetodict(df):
        df=df.fillna(0)
        re = {}
        colnames = df.columns.values.tolist()
        scaler = MinMaxScaler()
        for colname in colnames:
            if colname=="ts_code" or "end_date" in colname:
                re[colname] = np.array(df[colname])
            else:
                re[colname] = scaler.fit_transform(np.array(df[colname]).reshape(-1,1))
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
    #shape会创建单个值（标量) 一个x对应84维特征


    # train_input_fn = tf.estimator.inputs.numpy_input_fn({"x": np.array(train_x)}, np.array(train_y), batch_size=100, num_epochs=None,
    #                                                     shuffle=True)
    #
    # eval_input_fn = tf.estimator.inputs.numpy_input_fn({"x": np.array(valid_x)}, np.array(valid_y),
    #                                                     num_epochs=1,
    #                                                     shuffle=False)

    def train_input_fn(features, labels, batch_size):
        """An input function for training"""
        # Convert the inputs to a Dataset.
        features=dataframetodict(features)
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))

        # Shuffle, repeat, and batch the examples.
        dataset = dataset.shuffle(100000).repeat().batch(batch_size)

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
        # model_dir="./model",
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
    print("预测结果：",list(predictions)[0])

    print('\nTest set accuracy: {accuracy:}\n'.format(**eval_result))

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)


