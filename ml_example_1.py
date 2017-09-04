import tensorflow as tf

a = tf.constant(4, name='constant_a')
b = tf.constant(8, name='constant_b')
c = tf.constant(13, name='constant_c')
d = tf.constant(2, name='constant_d')

add = tf.add(a, b, name='sum_value')
mul = tf.multiply(c, d, name='product_value')

final = tf.add_n([add, mul], name='final_value')

sess = tf.Session()

print(sess.run(add))
print(sess.run(mul))
print(sess.run(final))

writer = tf.summary.FileWriter('./m2_ex_1', sess.graph)
writer.close()

sess.close()
