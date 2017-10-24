import tensorflow as tf
import numpy as np


def basic_nodes():
    """
    A tensor is a set of primitive values shaped into an
    array of any numpyr of dimensions.
    A tensor's rank is it's number of dimensions
    """
    t1=[1, 2, 3] # a rank one tesnor. vector with shape [3]
    t2=[[1,2,3],[4,5,6]] # a rank 2 tensor. matrix shape [2,3]
    t3=[[[1,2,3]],[[4,5,6]]] # a rank 3 tensor. tensor shape [2,1,3]

    """
    The Computational Graph: 2 sections
    1. Building the computational graph (compiling?)
    2. Running the computational graph

    The graph is a series of tf operations arranged into  graph
    of nodes.

    Each node takes zero or more tensors as inputs and produces
    a tensor as an output

    e.g. a tf constant node
    """
    node1 = tf.constant(3.0, dtype=tf.float32)
    node2 = tf.constant(4.0) #implicitly tf.float32
    #print(node1,node2)

    #Doesn't output 3.0 and 4.0, need to be in a session

    sess = tf.Session()
    print(sess.run([node1, node2]))

    node3 = tf.add(node1, node2)
    print(sess.run(node3))

    """
    A graph can be parameterized to accept external inputs, known as placeholders.
    A placeholder is a promise to provide a value later.
    """

    a = tf.placeholder(tf.float32)
    b = tf.placeholder(tf.float32)
    adder_node = a + b
    #sort of acts like a function

    #run using "feed_dict" argument
    print(sess.run(adder_node, {a: 3, b: 4.5}))
    print(sess.run(adder_node, {a:t3, b:t3}))

    add_and_triple = adder_node * 3
    print(sess.run(add_and_triple, {a:4, b:6}))

def nodes_with_variables():
    '''
    Variables allow us to add trainable parameters to a graph
    '''
    W = tf.Variable([.3], dtype=tf.float32) # weights?
    b = tf.Variable([-.3], dtype=tf.float32)# intercept?
    x = tf.placeholder(tf.float32)          # data?

    linear_model = W * x + b

    '''
    Constants are initialized with tf.constant and are const..
    Variables are not initialized by tf.Variable...
    Call ...
    '''
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    '''
    As X is a placeholder, can run several values of x simultaneously
    Like you would in a nn.....
    '''
    print(sess.run(linear_model, {x: [1,2,3,4]}))

    y = tf.placeholder(tf.float32)
    squared_deltas = tf.square(linear_model - y)
    # or
    #squared_deltas = (linear_model - y)**2
    #loss = tf.reduce_sum(squared_deltas)
    loss = tf.reduce_sum(squared_deltas)
    print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))

    fixW = tf.assign(W, [-1.])
    fixb = tf.assign(b, [1.])

    sess.run([fixW, fixb])
    print(sess.run(loss, {x:[1,2,3,4],y:[-0,-1,-2,-3]}))



if __name__ == "__main__":
    #basic_nodes()
    nodes_with_variables()
