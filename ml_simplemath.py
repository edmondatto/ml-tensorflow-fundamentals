import tensorflow as tf

a = tf.constant(3.8, name='constant_a')
b = tf.constant(2.7, name='constant_b')
c = tf.constant(10.0, name='constant_c')
d = tf.constant(8.9, name='constant_d')

poww = tf.pow(a, b, name='pow')
square = tf.square(c, name='square')
sqrt = tf.sqrt(d, name='square_root')

primary_value = tf.add_n([poww, square, sqrt], name='primary_value')

square_2 = tf.square(primary_value, name='loopy_square')

secondary_value = tf.add_n([square_2, poww, sqrt], name='secondary_value')

sess = tf.Session()
print(sess.run(primary_value))
print(sess.run(secondary_value))

writer = tf.summary.FileWriter('./ml_ex_2', sess.graph)
writer.close()

sess.close()
