import numpy as np
import matplotlib.pylab as plt
from matplotlib import cm
import chainer
from chainer import Chain, optimizers, training
from chainer.training import extensions
import chainer.functions as F
import chainer.links as L
from sklearn.datasets import fetch_mldata

# 畳み込みニューラルネットワークでMNIST画像分類 ver. classifierクラス, trainingクラス

# モデルクラス定義

class CNN(Chain):
    def __init__(self):
        # クラスの初期化
        super(CNN, self).__init__(            
            conv1 = L.Convolution2D(1, 20, 5), # フィルター5
            conv2 = L.Convolution2D(20, 50, 5), # フィルター5
            l1 = L.Linear(800, 500),
            l2 = L.Linear(500, 500),
            l3 = L.Linear(500, 10, initialW=np.zeros((10, 500), dtype=np.float32))
        )

    def __call__(self, x):
        # 順伝播の計算を行う関数
        # :param x: 入力値
        h = F.max_pooling_2d(F.relu(self.conv1(x)), 2)
        h = F.max_pooling_2d(F.relu(self.conv2(h)), 2)
        h = F.relu(self.l1(h))
        h = F.relu(self.l2(h))
        y = self.l3(h)
        return y

# 学習

EPOCH_NUM = 5
BATCH_SIZE = 1000

# 教師データ
mnist = fetch_mldata('MNIST original', data_home='.')
mnist.data = mnist.data.astype(np.float32) # 画像データ　784*70000 [[0-255, 0-255, ...], [0-255, 0-255, ...], ... ]
mnist.data /= 255 # 0-1に正規化する
mnist.target = mnist.target.astype(np.int32) # ラベルデータ70000

# 教師データを変換
dataset = []
for x, t in zip(mnist.data, mnist.target):
    dataset.append((x.reshape(1, 28, 28), t))
N = len(dataset)

# モデルの定義
model = L.Classifier(CNN())
optimizer = optimizers.Adam()
optimizer.setup(model)

# 学習開始
print("Train")
train, test = chainer.datasets.split_dataset_random(dataset, N-10000) # 60000件を学習用、10000件をテスト用
train_iter = chainer.iterators.SerialIterator(train, BATCH_SIZE)
test_iter = chainer.iterators.SerialIterator(test, BATCH_SIZE, repeat=False, shuffle=False)
updater = training.StandardUpdater(train_iter, optimizer, device=-1)
trainer = training.Trainer(updater, (EPOCH_NUM, "epoch"), out="result")
trainer.extend(extensions.Evaluator(test_iter, model, device=-1))
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport( ["epoch", "main/loss", "validation/main/loss", "main/accuracy", "validation/main/accuracy", "elapsed_time"])) # エポック、学習損失、テスト損失、学習正解率、テスト正解率、経過時間
#trainer.extend(extensions.ProgressBar()) # プログレスバー出力
trainer.run()

# 予測

print("\nPredict")
def predict(model, x):
    y = np.argmax(model.predictor(x=np.array([x], dtype="float32")).data)
    plt.figure(figsize=(1, 1))
    plt.imshow(x[0], cmap=cm.gray_r)
    plt.show()
    print("y:\t{}\n".format(y))

idx = np.random.choice(70000, 10)
for i in idx:
    predict(model, dataset[i][0])
