import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def import_data():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    train, test, val = mnist.train, mnist.test, mnist.validation
    return train, test, val

def y_graph():
    k = 784
    c = 10
    x = tf.placeholder(tf.float32, [None, k],'x')
    W = tf.Variable(tf.zeros([k,c]))
    b = tf.Variable(tf.zeros([c]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    return y, x

def cross_entropy_graph(y, y_):
    return tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y), reduction_indices=[1]))

def run_model(cost, x, y_, train, test, epochs, batch_size):
    train_step = tf.train.AdamOptimizer().minimize(cost)
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    while train.epochs_completed < epochs:
        b_xs, b_ys = train.next_batch(batch_size)
        sess.run(train_step, feed_dict={x:b_xs, y_:b_ys})
    print(train.epochs_completed)

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    xs, ys = test.images, test.labels
    print(sess.run(accuracy, feed_dict={x:xs, y_:ys}))

if __name__ == "__main__":
    train, test, val = import_data()
    y, x = y_graph()
    y_ = tf.placeholder(tf.float32, [None,10],'y_')

    cost = cross_entropy_graph(y,y_)
    run_model(cost, x, y_, train, test, epochs=10, batch_size=100)
