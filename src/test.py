import tensorflow as tf
from  tensorflow.examples.tutorials.mnist import input_data
import numpy as np

a = np.array([[1, 2], [3, 4]])
b = tf.one_hot(a, 4, axis=2)
with tf.Session() as sess:
    print(sess.run(b))