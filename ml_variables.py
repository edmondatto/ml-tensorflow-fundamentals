import tensorflow as tf

# y = Wx + b

W = tf.Variable([2.0, 3.5], tf.float32, name='var_W')
x = tf.placeholder(tf.float32, name='x')
b = tf.Variable([1, 4.4], tf.float32, name='var_b')

y = W * x + b

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    result = sess.run(y, feed_dict={x: [10, 100]})
    print('Final Result Is: {}'.format(result))

# Calculating intermediate values
s = W * x
init = tf.variables_initializer([W])

with tf.Session() as sess:
    sess.run(init)

    result = sess.run(s, feed_dict={x: [10, 100]})
    print('Final Result: {}'.format(result))

number = tf.Variable(2)
multiplier = tf.Variable(6)

init = tf.global_variables_initializer()
result = number.assign(tf.multiply(number, multiplier))

with tf.Session() as sess:
    sess.run(init)
    for i in range(10):
        print(sess.run(result))
        print('New Number Value: {}\n'.format(sess.run(number.assign_add(1))))

