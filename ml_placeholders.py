import tensorflow as tf

x = tf.placeholder(tf.int32, shape=[3], name='x')
y = tf.placeholder(tf.int32, shape=[3], name='y')

sum_x = tf.reduce_sum(x, name='sum_x')
prod_y = tf.reduce_prod(y, name='prod_y')
mean = tf.reduce_mean([sum_x, prod_y], name='mean')

sess = tf.Session()

sum_sess = sess.run(sum_x, feed_dict={x: [100, 200, 300]})
print(sum_sess)

prod_sess = sess.run(prod_y, feed_dict={y: [1, 2, 3]})
print(prod_sess)

mean_sess = sess.run(mean, feed_dict={x: [100, 200, 300], y: [1, 2, 3]})
print(mean_sess)

writer = tf.summary.FileWriter('./m2_ex_4', sess.graph)
writer.close()

sess.close()

