import tensorflow as tf

W = tf.Variable([.3], tf.float32, name="var_w")
b = tf.Variable([-.3], tf.float32, name="var_b")

# Model Input and Output
x = tf.placeholder(tf.float32, name='x')
linear_model = W * x + b

y = tf.placeholder(tf.float32)

# loss
loss = tf.reduce_sum(tf.square(linear_model - y))

# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# training data
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for i in range(1000):
        sess.run(train, {x: x_train, y: y_train})

    # evaluate training accuracy
    curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
    print("W: {}, b: {}, loss: {}".format(curr_W, curr_b, curr_loss))
