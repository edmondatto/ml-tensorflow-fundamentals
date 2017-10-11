import tensorflow as tf

g1 = tf.Graph()

with g1.as_default():
    with tf.Session() as sess:
        # y = Ax + b
        A = tf.constant([4, 3], tf.int32, name='const_A')
        x = tf.placeholder(tf.int32, name='x')
        b = tf.constant([8, 6], tf.int32, name='const_b')

        y = A * x + b
        result = sess.run(y, feed_dict={x: [10, 20]})
        print(result)
        assert y.graph is g1

g2 = tf.Graph()

with g2.as_default():
    with tf.Session() as sess:
        # y = A ^ x
        A = tf.constant([1, 3], tf.int32, name='const_a')
        x = tf.placeholder(tf.int32, name='x')
        y = tf.pow(A, x, name='y')

        result = sess.run(y, feed_dict={x: [2, 4]})
        print(result)
        assert y.graph is g2

default_graph = tf.get_default_graph()
with tf.Session() as sess:
    # y = A + x
    A = tf.constant([1, 3], tf.int32, name='const_a')
    x = tf.placeholder(tf.int32, name='x')
    y = tf.add(A, x, name='y')

    result = sess.run(y, feed_dict={x: [2, 4]})
    print(result)
    assert y.graph is default_graph
