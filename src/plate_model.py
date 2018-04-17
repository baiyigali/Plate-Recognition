import tensorflow as tf
import tensorflow.contrib.slim as slim
import cv2
import numpy as np
import data_process.label as data_label


class plate_Net(object):
    def __init__(self, images, batch_size, num_classes=10):
        self.images = images
        self.batch_size = batch_size
        self.num_classes = num_classes
        pass

    # input size [None, 96, 33, 3]
    # output size [None, num_classes, 1, 8]
    def network_model(self, alpha=0.1, scope='fcn'):
        with tf.variable_scope(scope):
            with slim.arg_scope([slim.conv2d, slim.fully_connected], activation_fn=self._leaky_relu(alpha),
                                weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                weights_regularizer=slim.l2_regularizer(0.0005)):
                net = self.images
                net = slim.conv2d(inputs=net, num_outputs=32, kernel_size=3, stride=1, padding='VALID', scope='conv_1')
                net = slim.max_pool2d(inputs=net, kernel_size=3, stride=2, padding='SAME', scope='pool_1')
                net = slim.conv2d(inputs=net, num_outputs=64, kernel_size=3, stride=1, padding='VALID', scope='conv_2')
                net = slim.max_pool2d(inputs=net, kernel_size=2, padding='SAME', scope='pool_2')
                net = slim.conv2d(inputs=net, num_outputs=128, kernel_size=3, stride=1, padding='SAME', scope='conv_3')
                net = slim.max_pool2d(inputs=net, kernel_size=2, padding='SAME', scope='pool_3')
                net = slim.conv2d(inputs=net, num_outputs=256, kernel_size=3, stride=1, padding='VALID', scope='conv_4')
                net = slim.conv2d(inputs=net, num_outputs=128, kernel_size=2, stride=1, padding='VALID', scope='conv_5')
                net = slim.conv2d(inputs=net, num_outputs=self.num_classes, kernel_size=1, stride=1, padding='VALID', scope='conv_6')
                net = slim.softmax(net, scope='output')
        return net

    def _leaky_relu(self, alpha):
        def op(inputs):
            return tf.maximum(alpha * inputs, inputs, name='leaky_relu')

    def loss(self, predicts, labels, scope='loss'):
        with tf.variable_scope(scope):
            loss = predicts - labels
            loss = tf.reduce_mean(tf.reduce_sum(tf.square(loss), axis=[1, 2, 3]), name='loss')
            tf.summary.scalar('loss', loss)
        return loss

    def accuracy(self, predicts, labels, scope='accuracy'):
        with tf.variable_scope(scope):
            prediction = tf.equal(tf.argmax(predicts, axis=3), tf.argmax(labels, axis=3))
            prediction = tf.cast(prediction, tf.float32)
            accuracy = tf.reduce_mean(prediction, name='accuracy')
            print(accuracy)
            tf.summary.scalar('accuracy', accuracy)
        return accuracy

# image = cv2.imread("./plate_image/sample.png")
# # image = cv2.resize(image, (100, 30))
# input = tf.placeholder(tf.float32, [None, 31, 94, 3])
# fcn = plate_Net(input, batch_size=32, num_classes=65)
# model = fcn.network_model(alpha=0.1)
# print(model)
