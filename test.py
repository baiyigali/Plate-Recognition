import random
import glob
import concurrent
import numpy as np
import tensorflow as tf
#
# array = np.random.random(size=32 * 8 * 82)
# array = array.reshape((32, 8, 82))
# print(array.shape)
# for i in array:
#     print(i.shape)
#     arg = np.argmax(i, 1)
#     print(arg.shape)

# A = [[1, 2], [3, 4]]
# B = [[2, 2], [3, 4]]
#
# with tf.Session() as sess:
#     print(sess.run(tf.equal(A, B)))

array1 = tf.random_normal((32, 8, 82))
array2 = tf.random_normal((32, 8, 82))
with tf.Session() as sess:
    class_delta = array1 - array2
    print(sess.run(tf.reduce_sum(tf.square(class_delta), axis=2)))
    # self.cost = tf.reduce_mean(tf.reduce_sum(tf.square(class_delta), axis=[1, 2, 3]))
