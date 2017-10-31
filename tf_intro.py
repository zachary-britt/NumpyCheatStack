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

def program():
    # Model parameters
    W = tf.Variable([.3], dtype=tf.float32)
    b = tf.Variable([-.3], dtype=tf.float32)
    # Model input and output
    x = tf.placeholder(tf.float32)
    linear_model = W * x + b
    y = tf.placeholder(tf.float32)

    # loss
    loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
    # optimizer
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)

    # training data
    x_train = [1, 2, 3, 4]
    y_train = [0, -1, -2, -3]
    # training loop
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init) # reset values to wrong
    for i in range(1000):
      sess.run(train, {x: x_train, y: y_train})

    # evaluate training accuracy
    curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
    print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))

def estimator():
    # Declare list of features
    feature_columns = [tf.feature_column.numeric_column('x', shape=[1])]

    # Linear regression estimator. A front end to do ML
    estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns)

    # t for training
    # e for evaluation/test
    x_t = np.array([1.,2.,3.,4.])
    y_t = np.array([0., -1., -2.,-3])
    x_e = np.array([2., 5., 8., 1.])
    y_e = np.array([-1.01, -4.1, -7, 0.])

    # helper function for read and setup
    # specify batches and epochs
    input_fn = tf.estimator.inputs.numpy_input_fn(
        {"x": x_t}, y_t, batch_size=4, num_epochs=None, shuffle=True)
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        {"x": x_t}, y_t, batch_size=4, num_epochs=1000, shuffle=False)
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        {"x": x_e}, y_e, batch_size=4, num_epochs=1000, shuffle=False)

    # invoke 1000 training steps
    estimator.train(input_fn=input_fn, steps=1000)

    # evaluate
    train_metrics = estimator.evaluate(input_fn=train_input_fn)
    eval_metrics = estimator.evaluate(input_fn=eval_input_fn)
    print("train metrics: %r"% train_metrics)
    print("eval metrics: %r"% eval_metrics)

def model_fn1(features, labels, mode):
    # build a linear model and predict values

    W = tf.get_variable("W", [1], dtype=tf.float64)
    b = tf.get_variable('b', [1], dtype=tf.float64)
    y = W * features['x'] + b

    # loss sub-graph
    loss = tf.reduce_sum(tf.square(y - labels))

    # training sub-graph
    global_step = tf.train.get_global_step()
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = tf.group(optimizer.minimize(loss),
                    tf.assign_add(global_step, 1))

    # connect subgraphs to appropriate functionality
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=y,
        loss=loss,
        train_op=train)

def custom_estimator1():
    estimator = tf.estimator.Estimator(model_fn=model_fn)
    x_t = np.array([1.,2.,3.,4.])
    y_t = np.array([0., -1., -2.,-3])
    x_e = np.array([2., 5., 8., 1.])
    y_e = np.array([-1.01, -4.1, -7, 0.])

    input_fn = tf.estimator.inputs.numpy_input_fn(
        {"x": x_t}, y_t, batch_size=4, num_epochs=None, shuffle=True)
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        {"x": x_t}, y_t, batch_size=4, num_epochs=1000, shuffle=False)
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        {"x": x_e}, y_e, batch_size=4, num_epochs=1000, shuffle=False)

    estimator.train(input_fn=input_fn, steps=1000)
    train_metrics = estimator.evaluate(input_fn=train_input_fn)
    eval_metrics = estimator.evaluate(input_fn)
    print("train metrics: %r"% train_metrics)
    print("eval metrics: %r"% eval_metrics)

# Declare list of features, we only have one real-valued feature
def model_fn(features, labels, mode):
    # Build a linear model and predict values
    W = tf.get_variable("W", [1], dtype=tf.float64)
    b = tf.get_variable("b", [1], dtype=tf.float64)
    y = W * features['x'] + b
    # Loss sub-graph
    loss = tf.reduce_sum(tf.square(y - labels))
    # Training sub-graph
    global_step = tf.train.get_global_step()
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = tf.group(optimizer.minimize(loss),
               tf.assign_add(global_step, 1))
    # EstimatorSpec connects subgraphs we built to the
    # appropriate functionality.
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=y,
        loss=loss,
        train_op=train)

def custom_estimator():
    estimator = tf.estimator.Estimator(model_fn=model_fn)
    # define our data sets
    x_train = np.array([1., 2., 3., 4.])
    y_train = np.array([0., -1., -2., -3.])
    x_eval = np.array([2., 5., 8., 1.])
    y_eval = np.array([-1.01, -4.1, -7, 0.])
    input_fn = tf.estimator.inputs.numpy_input_fn(
        {"x": x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True)
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        {"x": x_train}, y_train, batch_size=4, num_epochs=1000, shuffle=False)
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        {"x": x_eval}, y_eval, batch_size=4, num_epochs=1000, shuffle=False)

    # train
    estimator.train(input_fn=input_fn, steps=1000)
    # Here we evaluate how well our model did.
    train_metrics = estimator.evaluate(input_fn=train_input_fn)
    eval_metrics = estimator.evaluate(input_fn=eval_input_fn)
    print("train metrics: {}".format(train_metrics))
    print("eval metrics: {}".format(eval_metrics))


if __name__ == "__main__":
    #basic_nodes()
    #nodes_with_variables()
    # program()
    # estimator()
    custom_estimator()
