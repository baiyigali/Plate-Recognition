import tensorflow as tf
import numpy as np


class FCN():
    def __init__(self, images, num_classes):
        self.images = images
        self.num_classes = num_classes
        self.network_model()
        pass

    def network_model(self):

        with tf.name_scope('block1'):
            x = self.images
            x = self._conv(name='conv_1', x=x, filter_size=3, input_channels=3, out_channels=16,
                           strides=self._stride_arr(2))
            x = tf.nn.max_pool(value=x, ksize=[1, 2, 2, 1], strides=self._stride_arr(1), padding='SAME')
            x = tf.nn.relu(x)

        with tf.name_scope('block2'):
            x = self._conv(name='conv_2', x=x, filter_size=3, input_channels=16, out_channels=32,
                           strides=self._stride_arr(2))
            x = tf.nn.max_pool(value=x, ksize=[1, 2, 2, 1], strides=self._stride_arr(1), padding='SAME')
            x = tf.nn.relu(x)

        with tf.name_scope('block3'):
            x = self._conv(name='conv_3', x=x, filter_size=3, input_channels=32, out_channels=64,
                           strides=self._stride_arr(2))
            x = tf.nn.max_pool(value=x, ksize=[1, 2, 2, 1], strides=self._stride_arr(1), padding='SAME')
            x = tf.nn.relu(x)

        with tf.name_scope('block4'):
            x = self._conv(name='conv_4', x=x, filter_size=3, input_channels=64, out_channels=128,
                           strides=self._stride_arr(2))
            x = tf.nn.max_pool(value=x, ksize=[1, 2, 2, 1], strides=self._stride_arr(1), padding='SAME')
            x = tf.nn.relu(x)

        shape = x.get_shape().as_list()
        dim = np.prod(shape[1:])
        x = tf.reshape(x, [-1, dim])

        with tf.variable_scope('logit'):
            flatten = self._fully_connected(x, 1024, name='fc')
            fc0 = self._fully_connected(flatten, self.num_classes, name='fc0')
            self.fc0 = tf.nn.softmax(fc0)
            fc1 = self._fully_connected(flatten, self.num_classes, name='fc1')
            self.fc1 = tf.nn.softmax(fc1)
            fc2 = self._fully_connected(flatten, self.num_classes, name='fc2')
            self.fc2 = tf.nn.softmax(fc2)
            fc3 = self._fully_connected(flatten, self.num_classes, name='fc3')
            self.fc3 = tf.nn.softmax(fc3)
            fc4 = self._fully_connected(flatten, self.num_classes, name='fc4')
            self.fc4 = tf.nn.softmax(fc4)
            fc5 = self._fully_connected(flatten, self.num_classes, name='fc5')
            self.fc5 = tf.nn.softmax(fc5)
            fc6 = self._fully_connected(flatten, self.num_classes, name='fc6')
            self.fc6 = tf.nn.softmax(fc6)
            fc7 = self._fully_connected(flatten, self.num_classes, name='fc7')
            self.fc7 = tf.nn.softmax(fc7)
            # fc = tf.concat([fc0, fc1, fc2, fc3, fc4, fc5, fc6, fc7], axis=1)
            # self.out = tf.nn.softmax(fc, name='output')

    # fc layer
    def _fully_connected(self, x, out_dim, name='fc'):
        # transform to 2D tensor, size:[N, -1]
        # x = tf.reshape(x, [self.hps.batch_size, -1])

        # param: w, avg random init, [-sqrt(3/dim), sqrt(3/dim)]*factor
        w = tf.get_variable(name + 'DW', [x.get_shape()[1], out_dim],
                            initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
        # parm: b, 0 init
        b = tf.get_variable(name + 'biases', [out_dim], initializer=tf.constant_initializer())
        # x * w + b
        return tf.nn.xw_plus_b(x, w, b, name=name)

    # 2D conv
    def _conv(self, name, x, filter_size, input_channels, out_channels, strides, padding='SAME'):
        with tf.variable_scope(name):
            n = filter_size * filter_size * out_channels
            # get or new conv filer, init with random
            kernel = tf.get_variable(
                'DW',
                [filter_size, filter_size, input_channels, out_channels],
                tf.float32,
                initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / n)))
            # conv
            return tf.nn.conv2d(x, kernel, strides, padding=padding)

    # transform step to tf.nn.conv2d needed
    def _stride_arr(self, stride):
        return [1, stride, stride, 1]

