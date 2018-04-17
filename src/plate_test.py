import os
import sys

sys.path.append("..")
sys.path.append("../..")
import time
import numpy as np
import tensorflow as tf
from datetime import datetime
from src import plate_model
from src.datagenerator import ImageDataGenerator
from src.utils import mkdirs
import data_process.label as data_label

class plateNet_test(object):
    def __init__(self, image_size, batch_size,
                 num_digit, num_classes, test_file, checkpoint_path,
                 relu_leakiness=0, is_restore=True, device_id='2'):
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_digit = num_digit
        self.num_classes = num_classes
        self.display_step = 20
        self.train_file = test_file
        self.checkpoint_path = checkpoint_path
        mkdirs(self.checkpoint_path)
        self.relu_leakiness = relu_leakiness
        if is_restore:
            ckpt = tf.train.get_checkpoint_state(self.checkpoint_path)
            self.restore_checkpoint = ckpt.model_checkpoint_path
        else:
            self.restore_checkpoint = ''
        os.environ['CUDA_VISIBLE_DEVICES'] = device_id
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        self.init_model()

    def init_model(self):
        self.x = tf.placeholder(tf.float32, [None, self.image_size[0], self.image_size[1], 3], name='input')
        self.y = tf.placeholder(tf.float32, [None, 1, self.num_digit, self.num_classes])
        self.model = plate_model.plate_Net(self.x, batch_size=self.batch_size, num_classes=self.num_classes)
        self.predict = self.model.network_model()
        self.accuracy = self.model.accuracy(self.predict, self.y)
        self.loss = self.model.loss(self.predict, self.y)

    def create_date(self):
        return ImageDataGenerator(self.train_file, scale_size=self.image_size, num_digit=self.num_digit,
                                  num_classes=self.num_classes)

    def label2plate(self, predicts, labels):
        predicts_arg = tf.argmax(predicts, axis=2)
        labels_arg = tf.argmax(labels, axis=2)
        pre_char = []
        lab_char = []
        for i in range(predicts_arg.shape[0]):
            pre_char.append(data_label.class_char[predicts_arg[i].eval()[0]])
            lab_char.append(data_label.class_char[labels_arg[i].eval()[0]])
        print(''.join(pre_char), ''.join(lab_char))

    def test(self):
        self.saver = tf.train.Saver()

        test_generator = self.create_date()
        # Get the number of training/validation steps per epoch
        test_batches_per_epoch = np.floor(test_generator.data_size / self.batch_size).astype(np.int16)

        # Start Tensorflow session

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
            sess.run(tf.global_variables_initializer())

            # Load the pretrained weights into the non-trainable layer
            # if restore_checkponit is '' use ariginal weights, else use checkponit
            if not self.restore_checkpoint == '':
                self.saver.restore(sess, self.restore_checkpoint)

            # Validate the model on the entire validation set
            print("{} Start test, test num batches: {}, total num: {}".format(datetime.now(), test_batches_per_epoch,
                                                                              test_batches_per_epoch * self.batch_size))
            v_loss = 0.
            v_acc = 0.
            count = 0
            t1 = time.time()
            for i in range(test_batches_per_epoch):
                batch_validx, batch_validy = test_generator.next_batch(self.batch_size)
                valid_loss, valid_acc, valid_out = sess.run([self.loss, self.accuracy, self.predict],
                                                            feed_dict={self.x: batch_validx, self.y: batch_validy})

                # for i in range(self.batch_size):
                #     # print(i)
                #     pre = valid_out[i]
                #     lab = batch_validy[i]
                #     predicts_arg = tf.argmax(pre, axis=2)
                #     labels_arg = tf.argmax(lab, axis=2)
                #     print(sess.run([tf.transpose(predicts_arg), tf.transpose(labels_arg)]))
                #         # pre_char = []
                #         # lab_char = []
                #         # for i in range(predicts_arg.shape[0]):
                #         #     print(sess.run(predicts_arg[i]))
                #         #     pre_char.append(data_label.class_char[predicts_arg[i].eval()[0]])
                #         #     lab_char.append(data_label.class_char[labels_arg[i].eval()[0]])
                #         # print(''.join(pre_char), ''.join(lab_char))

                v_loss += valid_loss
                v_acc += valid_acc
                count += 1

            v_loss /= count
            v_acc /= count
            t2 = time.time() - t1
            print("Test loss = {:.4f}, acc = {:.4f}".format(v_loss, v_acc))
            print("Test image: {:.4f}ms per image, total time: {:4f}s".format(
                (t2 * 1000 / (test_batches_per_epoch * self.batch_size)), t2))

            # Reset the file pointer of the image data generator
            test_generator.reset_pointer()


if __name__ == '__main__':
    # train_file = "./path/license/train.txt"
    test_file = "./path/license/test.txt"

    # train_file = "./path/plate_process_image/train.txt"
    # test_file = "./path/plate_process_image/test.txt"
    #
    # train_file = "./path/plate_process_image_with_shear/train.txt"
    # valid_file = "./path/plate_process_image_with_shear/valid.txt"
    #
    # train_file = "./path/palte_process_image_without_shape/train.txt"
    # valid_file = "./path/palte_process_image_without_shape/valid.txt"

    # valid_file = "./path/plate_process_image_with_blockmove/valid.txt"

    pt = plateNet_test(image_size=(31, 94),
                       batch_size=64,
                       num_digit=8,
                       num_classes=65,
                       test_file=test_file,
                       checkpoint_path="./tmp/platenet/smooth_checkpoints",
                       relu_leakiness=0,
                       is_restore=True,
                       device_id='2,3'
                       )
    pt.test()
