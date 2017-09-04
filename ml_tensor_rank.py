import tensorflow as tf

sess = tf.Session()
zeroD = tf.constant(4)
oneD = tf.constant(['hello', 'tensorflow', 'world'])
twoD = tf.constant([['keep'], ['calm']])
threeD = tf.constant([[[1]], [[4]]])

print(sess.run(tf.rank(zeroD)))
print(sess.run(tf.rank(oneD)))
print(sess.run(tf.rank(twoD)))
print(sess.run(tf.rank(threeD)))

sess.close()