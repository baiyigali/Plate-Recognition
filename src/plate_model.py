import tensorflow as tf
import tensorflow.contrib.slim as slim
import cv2
import numpy as np
import data_process.label as data_label


class plate_Net(object):
    def __init__(self, images, batch_size, weight_decay=0.05, is_train=True, num_classes=10):
        self.images = images
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.is_train = is_train
        self.weight_decay = weight_decay
        pass

    # input size [None, 96, 33, 3]
    # output size [None, num_classes, 1, 8]
    def network_model(self, alpha=0.1, scope='fcn'):
        with tf.variable_scope(scope):
            with slim.arg_scope([slim.conv2d, slim.fully_connected], activation_fn=self._leaky_relu(alpha),
                                weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                weights_regularizer=slim.l2_regularizer(0.0005)):
                with slim.arg_scope([slim.batch_norm], is_training=self.is_train):
                    net = self.images
                    net = slim.conv2d(inputs=net, num_outputs=32, kernel_size=3, stride=1, padding='VALID', scope='conv_1')
                    net = slim.batch_norm(inputs=net, scope='batch_1')
                    net = slim.max_pool2d(inputs=net, kernel_size=3, stride=2, padding='SAME', scope='pool_1')
                    net = slim.conv2d(inputs=net, num_outputs=64, kernel_size=3, stride=1, padding='VALID', scope='conv_2')
                    net = slim.batch_norm(inputs=net, scope='batch_2')
                    net = slim.max_pool2d(inputs=net, kernel_size=2, padding='SAME', scope='pool_2')
                    net = slim.conv2d(inputs=net, num_outputs=128, kernel_size=3, stride=1, padding='SAME', scope='conv_3')
                    net = slim.batch_norm(inputs=net, scope='batch_3')
                    net = slim.max_pool2d(inputs=net, kernel_size=2, padding='SAME', scope='pool_3')
                    net = slim.conv2d(inputs=net, num_outputs=256, kernel_size=3, stride=1, padding='VALID', scope='conv_4')
                    net = slim.conv2d(inputs=net, num_outputs=128, kernel_size=2, stride=1, padding='VALID', scope='conv_5')
                    net = slim.conv2d(inputs=net, num_outputs=self.num_classes, kernel_size=1, stride=1, padding='VALID', scope='conv_6')
                    net = slim.softmax(net, scope='output')
        return net

    def network_model2(self, scope='fcn'):
        with tf.variable_scope(scope):
            net = self.images
            net = tf.layers.conv2d(net, filters=32, kernel_size=3, strides=1, padding='valid',activation=tf.nn.relu)
            # net = tf.layers.batch_normalization(net, training=self.is_train)
            # net = tf.nn.relu(net)
            net = tf.layers.max_pooling2d(net, pool_size=3, strides=2, padding='same')
            net = tf.layers.conv2d(net, filters=64, kernel_size=3, strides=1, padding='valid')
            # net = tf.layers.batch_normalization(net, training=self.is_train)
            net = tf.nn.relu(net)
            net = tf.layers.max_pooling2d(net, pool_size=2, strides=2, padding='same')
            net = tf.layers.conv2d(net, filters=128, kernel_size=3, strides=1, padding='same')
            # net = tf.layers.batch_normalization(net, training=self.is_train)
            net = tf.nn.relu(net)
            net = tf.layers.max_pooling2d(net, pool_size=2, strides=2, padding='same')
            net = tf.layers.conv2d(net, filters=256, kernel_size=3, strides=1, padding='valid')
            net = tf.layers.conv2d(net, filters=128, kernel_size=2, strides=1, padding='valid')
            net = tf.layers.conv2d(net, filters=self.num_classes, kernel_size=1, strides=1, padding='valid')
            net = tf.nn.softmax(net)

        return net


    #
    def _leaky_relu(self, alpha):
        def op(inputs):
            return tf.maximum(alpha * inputs, inputs, name='leaky_relu')

    def loss(self, predicts, labels, scope='loss'):
        with tf.variable_scope(scope):
            var_list = tf.global_variables()
            weight_var = [val for val in var_list if 'weights' in val.name]
            for w_var in weight_var:
                tf.add_to_collection(tf.GraphKeys.WEIGHTS, w_var)
            regularizer = tf.contrib.layers.l2_regularizer(scale=self.weight_decay)
            reg_term = tf.contrib.layers.apply_regularization(regularizer)

            loss = predicts - labels
            loss = tf.reduce_mean(tf.reduce_sum(tf.square(loss), axis=[1, 2, 3]), name='loss')
            tf.summary.scalar('loss', loss)
        return loss + reg_term

    def accuracy(self, predicts, labels, scope='accuracy'):
        with tf.variable_scope(scope):
            prediction = tf.equal(tf.argmax(predicts, axis=3), tf.argmax(labels, axis=3))
            prediction = tf.cast(prediction, tf.float32)
            accuracy = tf.reduce_mean(prediction, name='accuracy')
            print(accuracy)
            tf.summary.scalar('accuracy', accuracy)
        return accuracy

if __name__ == "__main__":
    # image = cv2.imread("./plate_image/sample.png")
    # # image = cv2.resize(image, (100, 30))
    input = tf.placeholder(tf.float32, [None, 31, 94, 3])
    fcn = plate_Net(input, batch_size=32, is_train=True, num_classes=65)
    # model = fcn.network_model2()
    model2 = fcn.network_model()
    print(model2)
