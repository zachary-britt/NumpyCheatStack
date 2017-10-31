import tensorflow as tf
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
import ipdb

def load_and_condition_MNIST_data():
    ''' loads and shapes MNIST image data '''
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train=X_train.reshape(-1,28*28)
    X_test=X_test.reshape(-1,28*28)

    # X_train_node = tf.constant(X_train, dtype=tf.float64)
    # X_test_node = tf.constant(X_test, dtype=tf.float64)

    y_train_ohe = np.zeros((y_train.shape[0],10))
    y_test_ohe = np.zeros((y_test.shape[0],10))

    y_train_ohe[np.arange(y_train.shape[0]),y_train]=1
    y_test_ohe[np.arange(y_test.shape[0]),y_test]=1

    # y_train_ohe_node=tf.constant(y_train_ohe, dtype=tf.int64)
    # y_test_ohe_node=tf.constant(y_test_ohe, dtype=tf.int64)

    return X_train, y_train_ohe, X_test, y_test_ohe



class Network:
    def __init__(self):
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)


class Layer:
    def __init__(self, inputs, units, activation):

        np_W = np.random.random(inputs*units).reshape(inputs,units)
        self.W = tf.Variable(np_W, dtype=tf.float64)

        np_b = np.random.random(units).reshape(1,-1)
        self.b = tf.Variable(np_b, dtype=tf.float64)

        self.x = tf.placeholder(tf.float64, (1,inputs))

        self.inputs=inputs
        self.units=units

        self.func=np.vectorize(activation)

    def forward_graph(self):
        #ipdb.set_trace()
        z_in = tf.add(tf.matmul(self.x, self.W), self.b)
        z_out = tf.float64
        f_graph = tf.py_func(self.func, [z_in], z_out, False)
        return f_graph

def relu(x):
    return max([0,x])

if __name__ == "__main__":
    #nn = Network()
    X_train, y_train, X_test, y_test  = load_and_condition_MNIST_data()
    input_dim=int(X_train.shape[1])
    l1 = Layer(input_dim, 124, relu)
    f1 = l1.forward_graph()
