import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD, RMSprop
from keras import backend as K

def error(y_true, y_pred):
    return K.sum(K.square(y_pred - y_true), axis=-1)

def build_model():
    model = Sequential()
    model.add(Dense(4, input_dim=1))
    model.add(Activation('sigmoid'))
    model.add(Dense(4))
    model.add(Activation('sigmoid'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    # rmsprop = RMSprop(lr=0.01, rho=0.9, epsilon=1e-06)
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss=error, optimizer=sgd)
    return model

seed = 123
np.random.seed(seed)

m = 50
x_train = np.linspace(-1, 1, m).reshape((m, 1))
y_train = np.abs(x_train)
model = build_model()
model.fit(x_train, y_train, nb_epoch=10000, batch_size=5)
y_hat = model.predict(x_train)

plt.scatter(x_train, y_train, color='r')
plt.scatter(x_train, y_hat, color='b')
# plt.savefig("mlp_approximate_abs_keras_10000.png")
plt.show()
