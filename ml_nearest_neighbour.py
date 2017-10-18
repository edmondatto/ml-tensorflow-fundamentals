import numpy as np
import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

# Store the MNIST data
mnist = input_data.read_data_sets("mnist_data/", one_hot=True)

training_digits, training_labels = mnist.train.next_batch(5000)
test_digits, test_labels = mnist.test.next_batch(300)

training_digit_pl = tf.placeholder("float", [None, 784])
test_digit_pl = tf.placeholder("float", [784])

# Nearest neighbour calculation using L1
l1_distance = tf.abs(tf.add(training_digit_pl, tf.negative(test_digit_pl)))
distance = tf.reduce_sum(l1_distance, axis=1)

# Prediction: Get the min distance index (Nearest Neighbour)
pred = tf.arg_min(distance, 0)

accuracy = 0

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    # loop over the test data
    for i in range(len(test_digits)):
        # Get nearest neighbour
        nn_index = sess.run(pred, feed_dict={training_digit_pl: training_digits, test_digit_pl: test_digits[i, :]})

        # Get nearest neighbour class label and compare it to the actual one
        print("Test", i, "Prediction:", np.argmax(training_labels[nn_index]), "True Label:", np.argmax(test_labels[i]))

        # Calculate Accuracy
        if np.argmax(training_labels[nn_index]) == np.argmax(np.argmax(test_labels[i])):
            accuracy += 1./len(test_digits)

    print("Done!")
    print("Accuracy: ", accuracy)