import numpy as np
import chainer
from chainer import Chain, optimizers, training
from chainer.training import extensions
import chainer.functions as F
import chainer.links as L

# ニューラルネットワークで排他的論理和回路のモデル ver. softmax, classifierクラス, trainingクラス

# モデルクラス定義

class NN(Chain):
    def __init__(self, in_size, hidden_size, out_size):
        # クラスの初期化
        # :param in_size: 入力層のサイズ
        # :param hidden_size: 隠れ層のサイズ
        # :param out_size: 出力層のサイズ
        super(NN, self).__init__(
            xh = L.Linear(in_size, hidden_size),
            hh = L.Linear(hidden_size, hidden_size),
            hy = L.Linear(hidden_size, out_size)
        )

    def __call__(self, x):
        # 順伝播の計算を行う関数
        # :param x: 入力値
        h = F.sigmoid(self.xh(x))
        h = F.sigmoid(self.hh(h))
        y = self.hy(h)
        return y

# 学習

EPOCH_NUM = 500
HIDDEN_SIZE = 5
BATCH_SIZE = 4

# 教師データ
train_data = [
    [[0, 0], [0]],
    [[1, 0], [1]],
    [[0, 1], [1]],
    [[1, 1], [0]]
]

# 教師データを変換
in_size = len(train_data[0][0]) #  入力サイズ
out_size = in_size # 出力サイズ
dataset = []
for i in range(25): # trainingのバリデーションチェック機能も確認したいので、100件に増やす
    for x, t in train_data:
        dataset.append((np.array(x, dtype="float32"), np.array(t[0], dtype="int32")))
N = len(dataset) # 教師データの総数

# モデルの定義
model = L.Classifier(NN(in_size=in_size, hidden_size=HIDDEN_SIZE, out_size=out_size))
optimizer = optimizers.Adam()
optimizer.setup(model)

# 学習開始
print("Train")
train, test = chainer.datasets.split_dataset_random(dataset, N-10) # 90件を学習用、10件をテスト用
train_iter = chainer.iterators.SerialIterator(train, BATCH_SIZE)
test_iter = chainer.iterators.SerialIterator(test, BATCH_SIZE, repeat=False, shuffle=False)
updater = training.StandardUpdater(train_iter, optimizer, device=-1)
trainer = training.Trainer(updater, (EPOCH_NUM, "epoch"), out="result")
trainer.extend(extensions.Evaluator(test_iter, model, device=-1))
trainer.extend(extensions.LogReport(trigger=(100, "epoch"))) # 100エポックごとにログ出力
trainer.extend(extensions.PrintReport( ["epoch", "main/loss", "validation/main/loss", "main/accuracy", "validation/main/accuracy", "elapsed_time"])) # エポック、学習損失、テスト損失、学習正解率、テスト正解率、経過時間
#trainer.extend(extensions.ProgressBar()) # プログレスバー出力
trainer.run()

# 予測

print("\nPredict")
def predict(model, x):
    y = np.argmax(model.predictor(x=np.array([x], dtype="float32")).data)
    print("x:\t{}\t =&gt; \ty:\t{}".format(x, y))

predict(model, [1, 0])
predict(model, [0, 0])
predict(model, [1, 1])
predict(model, [0, 1])
