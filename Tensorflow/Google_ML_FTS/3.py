import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot
from pandas.plotting import scatter_matrix
# ダウンロードしてきたやつ
INDEIES = ["N225",  # Nikkei 225, Japan
           "HSI",   # Hang Seng, Hong Kong
           "GDAXI", # DAX, German
           "DJI",   # Dow, US
           "GSPC"]  # S&P 500, US
'''
           "SSEC",  # Shanghai Composite Index (China)
           "BVSP"]  # BOVESPA, Brazil
'''
def tf_confusion_metrics(model, actual_classes, session, feed_dict):
    predictions = tf.argmax(model, 1)
    actuals = tf.argmax(actual_classes, 1)

    ones_like_actuals = tf.ones_like(actuals)
    zeros_like_actuals = tf.zeros_like(actuals)
    ones_like_predictions = tf.ones_like(predictions)
    zeros_like_predictions = tf.zeros_like(predictions)

    tp_op = tf.reduce_sum(
      tf.cast(
        tf.logical_and(
          tf.equal(actuals, ones_like_actuals),
          tf.equal(predictions, ones_like_predictions)
        ),
        "float"
      )
    )

    tn_op = tf.reduce_sum(
      tf.cast(
        tf.logical_and(
          tf.equal(actuals, zeros_like_actuals),
          tf.equal(predictions, zeros_like_predictions)
        ),
        "float"
      )
    )

    fp_op = tf.reduce_sum(
      tf.cast(
        tf.logical_and(
          tf.equal(actuals, zeros_like_actuals),
          tf.equal(predictions, ones_like_predictions)
        ),
        "float"
      )
    )

    fn_op = tf.reduce_sum(
      tf.cast(
        tf.logical_and(
          tf.equal(actuals, ones_like_actuals),
          tf.equal(predictions, zeros_like_predictions)
        ),
        "float"
      )
    )

    tp, tn, fp, fn = \
      session.run(
        [tp_op, tn_op, fp_op, fn_op],
        feed_dict
      )

    tpr = float(tp)/(float(tp) + float(fn))
    fpr = float(fp)/(float(tp) + float(fn))

    accuracy = (float(tp) + float(tn))/(float(tp) + float(fp) + float(fn) + float(tn))

    recall = tpr
    precision = float(tp)/(float(tp) + float(fp))

    f1_score = (2 * (precision * recall)) / (precision + recall)

    print('Precision = ', precision)
    print('Recall = ', recall)
    print('F1 Score = ', f1_score)
    print('Accuracy = ', accuracy)
def study():
    closing = pd.DataFrame()
    for index in INDEIES:
        # na_valuesは文字列"null"のとき空として扱う CSVみるとnullって書いてあります。
        df = pd.read_csv("./data/" + index + ".csv",na_values=["null"])
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
        closing[index] = df["Close"]
    #空の部分は古いので埋める。
    closing = closing.fillna(method="ffill")
    for index in INDEIES:
        closing[index] = closing[index] / max(closing[index])
        closing[index] = np.log(closing[index] / closing[index].shift())

    closing["positive"] = 0
    #closing["N225"] >= 0の行のpositiveに1をいれる。
    closing.ix[closing["N225"] >= 0, "positive"] = 1
    closing["negative"] = 0
    #closing["N225"] < 0の行のnegativeに1をいれる。
    closing.ix[closing["N225"] < 0, "negative"] = 1

    #1~3日前のデータを予測に使う
    days_before = range(1,4)

    columns = ["positive", "negative"]
    for i in days_before :
        columns += [index + "_" + str(i) for index in INDEIES]

    training = pd.DataFrame(
        # ['positive', 'negative', 'N225_1', 'HSI_1', 'GDAXI_1', 'DJI_1', 'GSPC_1', 'SSEC_1', 'BVSP_1', 'N225_2', 'HSI_2', 'GDAXI_2', 'DJI_2', 'GSPC_2', 'SSEC_2', 'BVSP_2', 'N225_3', 'HSI_3', 'GDAXI_3', 'DJI_3', 'GSPC_3', 'SSEC_3', 'BVSP_3']
        columns = columns
    )

    #なんで7から？
    for i in range(7, len(closing)):
        data = {}
        #予測の部分は当日のデータで
        data["positive"] = closing["positive"].ix[i]
        data["negative"] = closing["negative"].ix[i]
        #ほかの指標は１個前のデータを使用する。
        for index in INDEIES:
            for before in days_before :
                data[index + "_" + str(before)] = closing[index].ix[i - before]
        training = training.append(data, ignore_index=True)
    #一応確認
    print(training.describe())

    #2行目からは予測する元のデータ
    predictors = training[training.columns[2:]]
    #2行目までは予測するべきデータ
    classes = training[training.columns[:2]]

    # 80%をトレーニングに使う
    training_size = int(len(training) * 0.8)

    training_predictors = predictors[:training_size]
    training_classes = classes[:training_size]
    test_predictors = predictors[training_size:]
    test_classes = classes[training_size:]

    # 予測する列の数
    num_predictors = len(training_predictors.columns)
    # 予測するべき列の数
    num_classes = len(training_classes.columns)

    session = tf.Session()
    feature = tf.placeholder(tf.float32, shape=(None, num_predictors))
    actual_classes = tf.placeholder(tf.float32,  shape=(None, num_classes))
    weights = tf.Variable(tf.truncated_normal([num_predictors, num_classes], stddev=0.0001))
    biases = tf.Variable(tf.ones([num_classes]))
    model = tf.nn.softmax(tf.matmul(feature, weights) + biases)
    cost = -tf.reduce_sum(actual_classes*tf.log(model))
    training_step = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)
    init = tf.global_variables_initializer()
    session.run(init)
    correct_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(actual_classes, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    actual_classes_param = training_classes.values.reshape(len(training_classes.values), 2)

    for i in range(1, 30001):
        session.run(
          training_step,
          feed_dict={
            feature: training_predictors.values,
            actual_classes: actual_classes_param
          }
        )
        if i%5000 == 0:
            print( i, session.run(
             accuracy,
             feed_dict={
               feature: training_predictors.values,
               actual_classes: actual_classes_param
             }
           ))
    feed_dict= {
      feature: test_predictors.values,
      actual_classes: test_classes.values.reshape(len(test_classes.values), 2)
    }

    tf_confusion_metrics(model, actual_classes, session, feed_dict)

    session2 = tf.Session()

    feature = tf.placeholder(tf.float32, shape=(None, num_predictors))
    actual_classes = tf.placeholder(tf.float32,  shape=(None, num_classes))

    weights1 = tf.Variable(tf.truncated_normal([num_predictors, 50], stddev=0.0001))
    biases1 = tf.Variable(tf.ones([50]))

    weights2 = tf.Variable(tf.truncated_normal([50, 25], stddev=0.0001))
    biases2 = tf.Variable(tf.ones([25]))

    weights3 = tf.Variable(tf.truncated_normal([25, 2], stddev=0.0001))
    biases3 = tf.Variable(tf.ones([2]))

    hidden_layer_1 = tf.nn.relu(tf.matmul(feature, weights1) + biases1)
    hidden_layer_2 = tf.nn.relu(tf.matmul(hidden_layer_1, weights2) + biases2)
    model = tf.nn.softmax(tf.matmul(hidden_layer_2, weights3) + biases3)

    cost = -tf.reduce_sum(actual_classes*tf.log(model))

    train_op1 = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)

    init = tf.global_variables_initializer()
    session2.run(init)

    correct_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(actual_classes, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    for i in range(1, 30001):
        session2.run(
            train_op1,
            feed_dict={
              feature: training_predictors.values,
              actual_classes: actual_classes_param
            }
          )
        if i%5000 == 0:
            print( i, session2.run(
              accuracy,
              feed_dict={
                feature: training_predictors.values,
                actual_classes: actual_classes_param
              }
            ))

    feed_dict= {
      feature: test_predictors.values,
      actual_classes: test_classes.values.reshape(len(test_classes.values), 2)
    }

    tf_confusion_metrics(model, actual_classes, session2, feed_dict)


if __name__ == "__main__":
    study()
