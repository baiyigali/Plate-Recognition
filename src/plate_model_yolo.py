import tensorflow as tf
import tensorflow.contrib.slim as slim
import cv2
import numpy as np
import data_process.label as data_label


class plate_Net(object):
    def __init__(self, images, batch_size, num_classes=10, keep_prob=0.1):
        self.images = images
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.keep_prob = keep_prob

        self.cell_size = (1, 8)
        self.box_per_cell = 1
        self.boundary1 = self.cell_size[0] * self.cell_size[1] * self.box_per_cell
        self.boundary2 = self.boundary1 + self.cell_size[0] * self.cell_size[1] * 4
        self.classes_scale = 1.
        self.coord_scale = 5.
        pass

    # input size [None, 96, 33, 3]
    # output size [None, num_classes, 1, 8]
    def network_model(self, num_output=560, alpha=0.1, scope='fcn'):
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
                net = slim.flatten(net, scope='flatten')
                net = slim.fully_connected(net, 900, scope='fc_1')
                net = slim.dropout(net, keep_prob=self.keep_prob, scope='dropout')
                net = slim.fully_connected(net, num_output, scope='fc_2')
                net = slim.softmax(net, scope='output')
        return net

    def _leaky_relu(self, alpha):
        def op(inputs):
            return tf.maximum(alpha * inputs, inputs, name='leaky_relu')

    def calc_iou(self, box1, box2, scope='iou'):
        with tf.variable_scope(scope):
            pass


    def loss(self, predicts, labels, scope='loss'):
        with tf.variable_scope(scope):
            predict_confidence = tf.reshape(predicts[:self.boundary1],
                                            [self.batch_size, self.cell_size[0], self.cell_size[1], self.box_per_cell])
            predict_boxes = tf.reshape(predicts[self.boundary1:self.boundary2],
                                     [self.batch_size, self.cell_size[0], self.cell_size[1], 4])
            predict_classes = tf.reshape(predicts[self.boundary2:],
                                       [self.batch_size, self.cell_size[0], self.cell_size[1], self.num_classes])

            labels_confidence = tf.reshape(labels[:, :, :, 0],
                                            [self.batch_size, self.cell_size[0], self.cell_size[1], self.box_per_cell])
            labels_boxes = tf.reshape(labels[:, :, :, 1:5],
                                     [self.batch_size, self.cell_size[0], self.cell_size[1], 4])
            labels_classes = tf.reshape(labels[:, :, :, 5:],
                                       [self.batch_size, self.cell_size[0], self.cell_size[1], self.num_classes])

            # object_loss

            # box_loss
            box_delte = labels_confidence * (predict_boxes - labels_boxes)
            box_loss = tf.reduce_mean(tf.reduce_sum(tf.square(box_delte), axis=[1, 2, 3])) * self.coord_scale

            # class_loss
            class_delte = labels_confidence * (predict_classes - labels_classes)
            class_loss = tf.reduce_mean(tf.reduce_sum(tf.square(class_delte), axis=[1, 2, 3])) * self.classes_scale

            tf.losses.add_loss(box_loss)
            tf.losses.add_loss(class_loss)

            tf.summary.scalar('box_loss', box_loss)
            tf.summary.scalar('class_loss', class_loss)

    def class_accuracy(self, predicts, labels, scope='class_accuracy'):
        predict_classes = tf.reshape(predicts[self.boundary2:],
                                     [self.batch_size, self.cell_size[0], self.cell_size[1], self.num_classes])
        labels_classes = tf.reshape(labels[:, :, :, 5:],
                                    [self.batch_size, self.cell_size[0], self.cell_size[1], self.num_classes])
        with tf.variable_scope(scope):
            prediction = tf.equal(tf.argmax(predict_classes, axis=3), tf.argmax(labels_classes, axis=3))
            prediction = tf.cast(prediction, tf.float32)
            accuracy = tf.reduce_mean(prediction, name='accuracy')
            print(accuracy)
            tf.summary.scalar('accuracy', accuracy)
        return accuracy

    def coord_accuracy(self, predicts, labels, scope='coord_accuracy'):
        pass


if __name__ == "__main__":
    # image = cv2.imread("./plate_image/sample.png")
    # # image = cv2.resize(image, (100, 30))
    input = tf.placeholder(tf.float32, [None, 31, 94, 3])
    fcn = plate_Net(input, batch_size=32, num_classes=65)
    model = fcn.network_model(alpha=0.1)
    print(model)
