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
from src.utils import mkdirs, deldirs
from data_process import create_plate2, plate_process, read_xml
import cv2

PROJECT_PATH = '/home1/fsb/project/LPR/Plate-Recognition'


class plateNet_train(object):
    def __init__(self, image_size, num_epoch, batch_size, learning_rate,
                 num_digit, num_classes, train_file, test_file, filewriter_path, checkpoint_path,
                 relu_leakiness=0.1, is_restore=True, restore_path=None, device_id='2'):
        self.image_size = image_size
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.train_file = train_file
        self.val_file = test_file
        self.train_batches_per_epoch = int(170000 / self.batch_size)
        self.valid_batches_per_epoch = int(30000 / self.batch_size)
        self.learning_rate = learning_rate
        self.num_digit = num_digit
        self.num_classes = num_classes
        self.display_step = 20
        self.filewriter_path = filewriter_path
        deldirs(self.filewriter_path)
        mkdirs(self.filewriter_path)
        self.checkpoint_path = checkpoint_path
        print(self.checkpoint_path)
        mkdirs(self.checkpoint_path)
        self.relu_leakiness = relu_leakiness
        if is_restore:
            if restore_path is None:
                print("restore path is none")
                sys.exit()
            ckpt = tf.train.get_checkpoint_state(restore_path)
            self.restore_checkpoint = ckpt.model_checkpoint_path
        else:
            self.restore_checkpoint = None
        os.environ['CUDA_VISIBLE_DEVICES'] = device_id

        self.init_model()

    def init_model(self):
        self.x = tf.placeholder(tf.float32, [None, self.image_size[0], self.image_size[1], 3], name='input')
        self.y = tf.placeholder(tf.float32, [None, 1, self.num_digit, self.num_classes])
        tf.add_to_collection('predict_network', self.y)
        self.model = plate_model.plate_Net(self.x, batch_size=self.batch_size, num_classes=self.num_classes)
        self.predict = self.model.network_model()
        self.accuracy = self.model.accuracy(self.predict, self.y)
        self.loss = self.model.loss(self.predict, self.y)
        self.train_op = self.optimize_model()
        self.merge_summary = tf.summary.merge_all()

    def optimize_model(self):
        self.variable_list = [v for v in tf.trainable_variables()]
        gradients = tf.gradients(self.loss, self.variable_list)
        gradients = list(zip(gradients, self.variable_list))
        with tf.variable_scope('optimize'):
            optimizer = tf.train.MomentumOptimizer(self.learning_rate, momentum=0.9)
            train_op = optimizer.apply_gradients(gradients)
        return train_op


    def train(self):
        self.writer = tf.summary.FileWriter(self.filewriter_path)
        self.saver = tf.train.Saver()


        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
            init = (tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init)
            self.writer.add_graph(sess.graph)
            # if restore_checkponit is '' use ariginal weights, else use checkponit
            if self.restore_checkpoint is not None:
                self.saver.restore(sess, self.restore_checkpoint)

            coordinator = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

            print("{} Start training...".format(datetime.now()))
            print("{} Open Tensorboard :tensorboard --logdir {} --host localhost --port 6066".format(datetime.now(),
                                                                                                     self.filewriter_path))
            # Loop over number of epochs
            for epoch in range(self.num_epoch):
                print("Epoch number: {}/{}".format(epoch + 1, self.num_epoch))
                for step in range(self.train_batches_per_epoch):
                    image_batch, label_batch, coord_batch = sess.run([train_images, train_labels, train_coords])

                    # print(image_batch.shape, label_batch.shape)

                    feed_dict = {self.x: image_batch, self.y: label_batch}

                    sess.run(self.train_op, feed_dict=feed_dict)
                    # print("Iter {}/{}".format(step * self.batch_size, self.train_batches_per_epoch * self.batch_size))

                    if step % self.display_step == 0:
                        loss, acc, s = sess.run([self.loss, self.accuracy, self.merge_summary], feed_dict=feed_dict)
                        self.writer.add_summary(s, epoch * self.train_batches_per_epoch + step)
                        print("Iter {}/{}, training mini-batch loss = {:.5f}, training accuracy = {:.5f}".format(
                            step * self.batch_size, self.train_batches_per_epoch * self.batch_size, loss, acc))

                # Validate the model on the entire validation set
                print("{} Start validation, valid num batches: {}, total num: {}".format(datetime.now(),
                                                                                         self.valid_batches_per_epoch,
                                                                                         self.valid_batches_per_epoch * self.batch_size))
                v_loss = 0.
                v_acc = 0.
                count = 0
                t1 = time.time()
                for i in range(self.valid_batches_per_epoch):
                    image_batch, label_batch, coord_batch = sess.run([valid_images, valid_labels, valid_coords])
                    feed_dict = {self.x: image_batch, self.y: label_batch}
                    valid_loss, valid_acc, valid_out = sess.run([self.loss, self.accuracy, self.predict],
                                                                feed_dict=feed_dict)

                    v_loss += valid_loss
                    v_acc += valid_acc
                    count += 1

                v_loss /= count
                v_acc /= count
                t2 = time.time() - t1
                print("Validation loss = {:.4f}, acc = {:.4f}".format(v_loss, v_acc))
                print("Test image {:.4f}ms per image".format(
                    t2 * 1000 / (self.valid_batches_per_epoch * self.batch_size)))

                print("{} Saving checkpoint of model...".format(datetime.now()))
                # save checkpoint of the model
                checkpoint_name = os.path.join(self.checkpoint_path, str(epoch + 1))
                self.saver.save(sess, checkpoint_name)

                print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))


if __name__ == '__main__':
    pt = plateNet_train(image_size=(31, 94),
                        num_epoch=20,
                        batch_size=200,
                        learning_rate=0.001,
                        num_digit=8,
                        num_classes=65,
                        filewriter_path=os.path.join(PROJECT_PATH, "tmp/platenet/tensorboard"),
                        checkpoint_path=os.path.join(PROJECT_PATH, "tmp/platenet/based_real_records_checkpoints"),
                        relu_leakiness=0.1,
                        is_restore=False,
                        restore_path=os.path.join(PROJECT_PATH, 'tmp/platenet/process_checkpoints'),
                        device_id='0'
                        )
    pt.train()
