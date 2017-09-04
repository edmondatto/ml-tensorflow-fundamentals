import tensorflow as tf

# y = Wx + b

w = tf.constant([10, 6], name="const_w")

x = tf.placeholder(tf.int32, name='x')
b = tf.placeholder(tf.int32, name='b')

wx = tf.multiply(w, x, name='wx')

y = tf.add_n([wx, b], name='y')

with tf.Session() as sess:
    print(sess.run(wx, feed_dict={x: [3, 4]}))
    print(sess.run(y, feed_dict={x: [6, 9], y: [2, 7]}))

writer = tf.summary.FileWriter('./m2_ex_5', sess.graph)
writer.close()
