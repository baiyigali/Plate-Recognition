import tensorflow as tf
import os
import numpy as np
import data_process.label as datalabel

os.environ['CUDA_VISIBLE_DEVICES'] = ''

array1 = tf.random_normal((32, 1, 8, 64))
array2 = tf.random_normal((32, 1, 8, 64))
with tf.Session() as sess:
    print(array1[0])
    arr = tf.argmax(array1[0], axis=2)
    print(arr)
    char = []
    print(sess.run([tf.transpose(arr), arr]))
    # for i in range(arr.shape[0]):
    #     sess.run(datalabel.class_char[arr[i][0]])
        # char.append(datalabel.class_char[arr[i].eval()[0]])
    # print(''.join(char))

    # for i in range(32):
    #     print(array1[i])
