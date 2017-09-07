import datetime
import numpy as np
from chainer import Chain, Variable, cuda, optimizer, optimizers, serializers
import chainer.functions as F
import chainer.links as L

# ニューラルネットワークで排他的論理和回路のモデル ver. softmax

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

    def __call__(self, x, t=None, train=False):
        # 順伝播の計算を行う関数
        # :param x: 入力値
        # :param t: 正解のラベル
        # :param train: 学習かどうか
        # :return: 計算した損失 or 予測したラベル
        x = Variable(x)
        if train:
            t = Variable(t)
        h = F.sigmoid(self.xh(x))
        h = F.sigmoid(self.hh(h))
        y = F.softmax(self.hy(h))
        if train:
            loss, accuracy = F.softmax_cross_entropy(y, t), F.accuracy(y, t)
            return loss, accuracy
        else:
            return np.argmax(y.data)

    def reset(self):
        # 勾配の初期化
        self.zerograds()

# 学習

EPOCH_NUM = 2000
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
N = len(train_data) # 教師データの総数
train_x, train_t = [], []
for x, t in train_data:
    train_x.append(x)
    train_t.append(t[0])
train_x = np.array(train_x, dtype="float32")
train_t = np.array(train_t, dtype="int32")

# モデルの定義
model = NN(in_size=in_size, hidden_size=HIDDEN_SIZE, out_size=out_size)
optimizer = optimizers.Adam()
optimizer.setup(model)

# 学習開始
print("Train")
st = datetime.datetime.now()
for epoch in range(EPOCH_NUM):
    # ミニバッチ学習
    perm = np.random.permutation(N) # ランダムな整数列リストを取得
    total_loss = 0
    total_accuracy = 0
    for i in range(0, N, BATCH_SIZE): 
        x = train_x[perm[i:i+BATCH_SIZE]]
        t = train_t[perm[i:i+BATCH_SIZE]]
        model.reset()
        loss, accuracy = model(x=x, t=t, train=True)
        loss.backward()
        loss.unchain_backward()
        total_loss += loss.data
        total_accuracy += accuracy.data
        optimizer.update()
    if (epoch+1) % 500 == 0:
        ed = datetime.datetime.now()
        print("epoch:\t{}\ttotal loss:\t{}\tmean accuracy:\t{}\ttime:\t{}".format(epoch+1, total_loss, total_accuracy/(N/BATCH_SIZE), ed-st))
        st = datetime.datetime.now()

# 予測

print("\nPredict")
def predict(model, x):
    y = model(x=np.array([x], dtype="float32"), train=False)
    print("x:\t{}\t =&gt; \ty:\t{}".format(x, y))

predict(model, [1, 0])
predict(model, [0, 0])
predict(model, [1, 1])
predict(model, [0, 1])
