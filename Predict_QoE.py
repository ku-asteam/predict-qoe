import csv
import numpy as np
import itertools
from sklearn.decomposition import PCA
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import json

tf.set_random_seed(777)
use_batchnorm = True

def get_file():
    # 데이터 불러오기 array[list]
    rawdata = []
    with open('csi/' + filename + '.csv', 'r') as raw:
        lines = csv.reader(raw)
        for line in lines:
            if line[1] != "0": #CSI가 누락된 경우를 제외하는 조건
                rawdata.append(line)
    rawdata = np.array(rawdata)
    return rawdata

# 18개씩 자르기
def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

def PreProcessing(rawdata):
    cleandata = []
    metadata = []

    for line in rawdata:
        metadata.append(line[7:10]) #num_tones, nr, nc 저장
        cleandata.append(line[15:(len(line) - int(line[14]))]) # 불필요한 데이터 자르기, 0~14, payload

    metadata = np.array(metadata)
    # nr * nc * num_tones * 2 랑 length 맞으면 정상
    for a, b in zip(cleandata, metadata):
        print(b[0], b[1], b[2], len(a))
        if int(b[0]) * int(b[1]) * int(b[2]) * 2 == len(a):
            print("True")
        else:
            print("False")

    # 2번
    SliceData = []
    for line in cleandata:
        SliceData.append(list(chunks(line, 18)))
    SliceData = np.array(SliceData)

    SliceData = np.transpose(SliceData, (1, 0, 2))

    Finaldata = []
    for line in SliceData:
        j = list(itertools.chain(*line))
        Finaldata.append(j)
    Finaldata = np.array(Finaldata).astype(np.float)


    pca = PCA(n_components=50)
    pca.fit(Finaldata)
    PCAdata = pca.transform(Finaldata)

    meanVector = np.mean(PCAdata, axis=0)
    return meanVector

def get_QoE(file_path, folder1):
    with open(file_path + '/' + folder1 + '.json', mode='r', encoding='UTF8') as f:
        json_string = json.load(f)
    data = json.dumps(json_string, indent=2)
    json_acceptable_string = data.replace("'", "\"")
    json_list = json.loads(json_acceptable_string)
    QoEData = json_list["visualMetrics"][0]["SpeedIndex"]
    return QoEData

if __name__ == "__main__":
    i = 1
    x_data = []
    while i < 37:
        filename = "res" + str(i)
        print("\n\n", filename + " Loading")
        PreData = PreProcessing(get_file())
        x_data.append(PreData)
        i = i + 1

    scaler = MinMaxScaler(feature_range=(0, 1000))
    x_data = scaler.fit_transform(x_data)
    print(x_data)
    print("\n\nx_data", x_data)

    y_data = np.array(get_QoE())

    scaler = MinMaxScaler(feature_range=(0, 100))
    y_data = scaler.fit_transform(y_data)
    print(y_data)

    trainingdata = x_data[:80/len(x_data)]
    label = y_data[:80/len(x_data)]

    testdata = x_data[20/len(x_data):]
    testlabel = y_data[20/len(x_data):]

    x = tf.placeholder(tf.float32, shape=[None, np.size(trainingdata, 1)])
    y = tf.placeholder(tf.float32, shape=[None, np.size(label, 1)])

    W1 = tf.get_variable("weight1", shape=[np.size(trainingdata, 1), 30],
                         initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.Variable(tf.random_normal([30]), name='bias1')
    hiddenlayer1 = tf.matmul(x, W1) + b1

    W2 = tf.Variable(tf.random_normal([30, 15]), name='weight2')
    b2 = tf.Variable(tf.random_normal([15]), name='bias2')
    hiddenlayer2 = tf.matmul(hiddenlayer1, W2) + b2

    W3 = tf.Variable(tf.random_normal([15, 10]), name='weight3')
    b3 = tf.Variable(tf.random_normal([10]), name='bias3')
    hiddenlayer3 = tf.matmul(hiddenlayer2, W3) + b3

    W4 = tf.Variable(tf.random_normal([10, 1]), name='weight4')
    b4 = tf.Variable(tf.random_normal([1]), name='bias4')
    hypothesis = tf.matmul(hiddenlayer3, W4) + b4

    cost = tf.reduce_mean(tf.square(hypothesis - y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.02)
    train = optimizer.minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    W_history = []
    cost_history = []
    for step in range(4000):
        _, cost_val, = sess.run([train, cost], feed_dict={x: trainingdata, y: label})
        if step % 200 == 0:
            print("\nStep: ", step, "\nCost: ", cost_val)
            W_history.append(step)
            cost_history.append(cost_val)

    predict_val = sess.run(hypothesis, feed_dict={x: trainingdata})
    differ1 = 0
    count1 = 0
    differ2 = 0
    print("------------------------------------------------training 데이터")
    for p, y in zip(predict_val, label):
        msg = "Prediction: {:d} \t True: {:d} \t[{}]\t differ: {:d} \t"
        differ1 = abs(int(p) - int(y[0]))
        print(msg.format(int(p), int(y[0]), int(p) == int(y[0]), differ1))
        if int(p) == int(y[0]):
            count1 = count1 + 1
        differ2 = differ1 + differ2

    print("Training Data", len(label), "개 / ", "Training Accuracy:", round(count1 / len(label) * 100), "%", "/ Total differ:", differ2)

    RMSE = mean_squared_error(label, predict_val)**0.5
    print("RMSE ", RMSE)

    predict_val = sess.run(hypothesis, feed_dict={x: testdata})
    scaler = MinMaxScaler(feature_range=(0, 100))
    predict_val = scaler.fit_transform(predict_val)
    differ1 = 0
    count1 = 0
    differ2 = 0
    print("------------------------------------------------test 데이터")
    for p, y in zip(predict_val, testlabel):
        msg = "Prediction y: {:d} \t True: {:d} \t[{}]\t differ: {:d} \t"
        differ1 = abs(int(p) - int(y[0]))
        print(msg.format(int(p), int(y[0]), int(p) == int(y[0]), differ1))
        if int(p) == int(y[0]):
            count1 = count1 + 1
        differ2 = differ1 + differ2

    print("Training Data", len(testlabel), "개 / ", "Training Accuracy:", round(count1 / len(testlabel) * 100), "%",
          "/ Total differ:", differ2)
    RMSE = mean_squared_error(testlabel, predict_val)**0.5
    print("RMSE ", RMSE)
    plt.plot(W_history, cost_history)
    plt.ylabel('Cost')
    plt.xlabel('Step')
    plt.show()
