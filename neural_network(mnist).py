import numpy as np
import matplotlib.pylab as plt
from sklearn.datasets import fetch_mldata
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Chain, optimizers, Variable, serializers

mnist = fetch_mldata('MNIST original', data_home='.')
mnist.data = mnist.data.astype(np.float32) # image data 784*70000 [[0-255, 0-255, ...], [0-255, 0-255, ...], ... ]
mnist.data /= 255 # to 0-1
mnist.target = mnist.target.astype(np.int32) # label data 70000
N = 60000
x_train, x_test = np.split(mnist.data,   [N])
t_train, t_test = np.split(mnist.target, [N])

class NN(Chain):
    def __init__(self, n_in, n_out):
        super(NN, self).__init__(
            l1 = L.Linear(n_in, 1000),
            l2 = L.Linear(1000, 1000),
            l3 = L.Linear(1000, 1000),
            l4 = L.Linear(1000, n_out, initialW=np.zeros((n_out, 1000), dtype=np.float32))
        )
    def forward(self, x):
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        h = F.relu(self.l3(h))
        h = self.l4(h)
        return h
    
model = NN(784, 10)
optimizer = optimizers.Adam()
optimizer.setup(model)

n_epoch = 5
batch_size = 1000
for epoch in range(n_epoch):
    sum_loss = 0
    sum_accuracy = 0
    perm = np.random.permutation(N)
    for i in range(0, N, batch_size):
        x = Variable(x_train[perm[i:i+batch_size]])
        t = Variable(t_train[perm[i:i+batch_size]])
        y = model.forward(x)        
        model.zerograds()
        loss = F.softmax_cross_entropy(y, t)
        acc = F.accuracy(y, t)
        loss.backward()
        optimizer.update()
        sum_loss += loss.data*batch_size
        sum_accuracy += acc.data*batch_size
    print("epoch: {}, mean loss: {}, mean accuracy: {}".format(epoch, sum_loss/N, sum_accuracy/N))
    
cnt = 0
for i in range(10000):
    x = Variable(np.array([x_test[i]], dtype=np.float32))
    t = t_test[i]
    y = model.forward(x)
    y = np.argmax(y.data[0])
    if t == y:
        cnt += 1

print("accuracy: {}".format(cnt/10000))
