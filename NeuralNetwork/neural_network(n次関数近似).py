"""
n次関数(y = 0.3x^2 + 0.2x + 0.1)の近似

"""
import tensorflow as tf
import numpy as np
import random
import math
import time

start = time.time()
AC=100
# number of W
WN=3

#1+number of the function s dimention
NN=3
a = np.arange(0.1,(WN+1)*0.1,0.1)

def y_from_x(_x,_W,_b):
    _y = np.dot(_W,_x) + _b
    return _y

x_data = np.random.rand(AC,NN,1).astype("float32")
y_data = np.zeros((AC,NN,1)).astype("float32")
npow = 1

for i in range(NN):
    y_data += a[i] * npow
    npow *= x_data

W = tf.Variable(tf.random_uniform([WN], -1.0, 1.0))
y = 0
npow = 1


for i in range(WN):
    y += W[i] * npow
    npow *= x_data

loss = tf.reduce_mean(tf.square(y_data - y))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in range(5001):
    sess.run(train)
    if step % 100 == 0:
        print( "training...", sess.run(W))

for i in range(100):
    result = 0
    npow = 1


    for j in range(WN):
        result += W[j] * npow
        npow *= i

    print(i, sess.run(result))

sess.close()

timer = time.time() - start

print ("time:{0}".format(timer) + "[sec]")
