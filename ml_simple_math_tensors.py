import tensorflow as tf

a = tf.constant([100, 200, 300], name='x')
b = tf.constant([1, 2, 3], name='y')

sum_x = tf.reduce_sum(a, name='sum_x')
prod_y = tf.reduce_prod(b, name='prod_y')

final_value = tf.div(sum_x, prod_y, name='final_value')
mean = tf.reduce_mean([sum_x, prod_y], name='mean')

sess = tf.Session()

print(sess.run(final_value))
print(sess.run(mean))

writer = tf.summary.FileWriter('./m2_ex_3', sess.graph)
writer.close()
sess.close()
