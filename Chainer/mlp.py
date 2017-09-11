import numpy as np
import chainer
from chainer import optimizers
import chainer.functions as F
import chainer.links as L
import matplotlib.pyplot as plt


class MLP(chainer.Chain):
    def __init__(self):
        super(MLP, self).__init__(
            l1=L.Linear(1, 4),
            l2=L.Linear(4, 4),
            l3=L.Linear(4, 1),
        )

    def __call__(self, x):
        h1 = F.sigmoid(self.l1(x))
        h2 = F.sigmoid(self.l2(h1))
        y = F.sigmoid(self.l3(h2))
        return y


class Regressor(chainer.Chain):
    def __init__(self, predictor):
        super(Regressor, self).__init__(predictor=predictor)

    def __call__(self, x, t):
        y = self.predictor(x)
        self.loss = F.sum((y - t)**2)
        return self.loss


if __name__ == '__main__':
    np.random.seed(123)
    m = 50
    x_train = np.linspace(-1, 1, m, dtype=np.float32).reshape((m, 1))
    y_train = np.abs(x_train)

    model = Regressor(MLP())
    optimizer = optimizers.SGD(lr=0.1)
    optimizer.setup(model)

    batchsize = 5
    for epoch in range(10000):
        indexes = np.random.permutation(m)
        sum_loss = 0.
        for i in range(0, m, batchsize):
            x = chainer.Variable(x_train[indexes[i: i + batchsize]])
            t = chainer.Variable(y_train[indexes[i: i + batchsize]])
            optimizer.update(model, x, t)
            sum_loss += model.loss.data
        print("epoch: {0:5d}, loss: {1:.5f}".format(epoch, sum_loss))

    y_hat = model.predictor(chainer.Variable(x_train))
    plt.scatter(x_train, y_train, color='r')
    plt.scatter(x_train, y_hat.data, color='b')
    # plt.savefig("mlp_approximate_abs_10000.png")
plt.show()
