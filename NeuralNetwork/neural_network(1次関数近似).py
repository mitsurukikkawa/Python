"""
1次関数(y = 0.1*x + 0.3 )の近似
"""
import tensorflow as tf
import numpy as np
import time

start = time.time()

x_data = np.random.rand(100).astype("float32")
y_data = 0.1 * x_data + 0.3

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

loss = tf.reduce_mean(tf.square(y_data - y))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in range(1001):
    sess.run(train)
    if step % 100 == 0:
        print(step, sess.run(W), sess.run(b))

for i in range(100):
    print(i, sess.run(W) * i + sess.run(b))

sess.close()

timer = time.time() - start
print (("time:{0}".format(timer)) + "[sec]")