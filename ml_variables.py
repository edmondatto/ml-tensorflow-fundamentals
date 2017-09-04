import tensorflow as tf

w = tf.Variable([2.4, 5.0], tf.float32, name='w')
x = tf.placeholder(tf.float32, name='x')
b = tf.Variable([9.0, 7.4], tf.float32, name='b')

y = w * x + b

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(y, feed_dict={x: [3, 5]}))

writer = tf.summary.FileWriter('./m2_ex_6', sess.graph)
writer.close()

