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
import concurrent.futures

PROJECT_PATH = '/home1/fsb/project/LPR/Plate-Recognition'


class plateNet_train(object):
    def __init__(self, image_size, num_epoch, batch_size, learning_rate,
                 num_digit, num_classes, filewriter_path, checkpoint_path,
                 relu_leakiness=0.1, is_restore=True, restore_path=None, device_id='2'):
        self.image_size = image_size
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_digit = num_digit
        self.num_classes = num_classes
        self.display_step = 20
        self.filewriter_path = filewriter_path
        self.step_per_epoch = 10000
        deldirs(self.filewriter_path)
        mkdirs(self.filewriter_path)
        self.checkpoint_path = checkpoint_path
        mkdirs(self.checkpoint_path)
        self.relu_leakiness = relu_leakiness
        print(restore_path)
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

    def get_variable_list(self):
        self.variable_list = [v for v in tf.trainable_variables()]
        return self.variable_list

    def optimize_model(self):
        self.variable_list = [v for v in tf.trainable_variables()]
        gradients = tf.gradients(self.loss, self.variable_list)
        gradients = list(zip(gradients, self.variable_list))
        with tf.variable_scope('optimize'):
            optimizer = tf.train.MomentumOptimizer(self.learning_rate, momentum=0.9)
            train_op = optimizer.apply_gradients(gradients)
        return train_op

    # generate plate data set in the memory
    def gen_data(self):
        batch_x = np.ndarray([self.batch_size, self.image_size[0], self.image_size[1], 3])
        batch_y = np.zeros((self.batch_size, 1, self.num_digit, self.num_classes))
        plate = create_plate2.plate()
        process = plate_process.plate_process()
        read_x = read_xml.read_xml(self.batch_size)

        serial_list = []
        for batch in range(self.batch_size):
            serial = plate.create_random_serial()
            serial_list.append(serial)

        def exe(serial):
            plate_image = plate.create_plate(serial).astype('float32')
            plate.imshow(plate_image)
            print(plate_image)
            plate_image = process.process_pic_with_shape_change(plate_image)

            label = read_x.serial2label(serial)
            return plate_image, label

        # with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            for i, (plate_image, plate_label) in enumerate(executor.map(exe, serial_list)):
                print(i)
                # plate_image -= np.array([127.5, 127.5, 127.5])
                # plate_image = cv2.resize(plate_image, (self.image_size[1], self.image_size[0]))
                # batch_x[i] = plate_image
                # batch_y[i] = plate_label

        return batch_x, batch_y

    def train(self):
        self.writer = tf.summary.FileWriter(self.filewriter_path)
        self.saver = tf.train.Saver()

        # Start Tensorflow session
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
            sess.run(tf.global_variables_initializer())

            self.writer.add_graph(sess.graph)

            # Load the pretrained weights into the non-trainable layer, if restore_checkponit is '' use ariginal weights, else use checkponit
            if self.restore_checkpoint is not None:
                self.saver.restore(sess, self.restore_checkpoint)

            print("{} Start training...".format(datetime.now()))
            print("{} Open Tensorboard :tensorboard --logdir {} --host localhost --port 6066".format(datetime.now(),
                                                                                                     self.filewriter_path))
            # Loop over number of epochs
            for epoch in range(self.num_epoch):

                print("Epoch number: {}/{}".format(epoch + 1, self.num_epoch))

                step = 1
                while step < self.step_per_epoch:
                    batch_xs, batch_ys = self.gen_data()
                    feed_dict = {self.x: batch_xs, self.y: batch_ys}

                    sess.run(self.train_op, feed_dict=feed_dict)

                    # Generate summary with the current batch of data and write to file
                    if step % self.display_step == 0:
                        batch_xs, batch_ys = self.gen_data()
                        feed_dict = {self.x: batch_xs, self.y: batch_ys}
                        loss, acc, s = sess.run([self.loss, self.accuracy, self.merge_summary], feed_dict=feed_dict)
                        self.writer.add_summary(s, epoch * self.step_per_epoch + step)
                        print("Iter {}/{}, training mini-batch loss = {:.5f}, training accuracy = {:.5f}".format(
                            step * self.batch_size, self.step_per_epoch * self.batch_size, loss, acc))

                    step += 1

                # Validate the model on the entire validation set
                print("{} Start validation, valid num batches: {}, total num: {}".format(datetime.now(),
                                                                                         self.step_per_epoch,
                                                                                         self.step_per_epoch * self.batch_size))
                v_loss = 0.
                v_acc = 0.
                count = 0
                t1 = time.time()
                for i in range(int(self.step_per_epoch / 10)):
                    batch_validx, batch_validy = self.gen_data()
                    valid_loss, valid_acc, valid_out = sess.run([self.loss, self.accuracy, self.predict],
                                                                feed_dict={self.x: batch_validx, self.y: batch_validy})

                    v_loss += valid_loss
                    v_acc += valid_acc
                    count += 1

                v_loss /= count
                v_acc /= count
                t2 = time.time() - t1
                print("Validation loss = {:.4f}, acc = {:.4f}".format(v_loss, v_acc))
                print("Test image {:.4f}ms per image".format(t2 * 1000 / (self.step_per_epoch / 10 * self.batch_size)))

                print("{} Saving checkpoint of model...".format(datetime.now()))
                # save checkpoint of the model
                checkpoint_name = os.path.join(self.checkpoint_path, str(epoch + 1))
                self.saver.save(sess, checkpoint_name)

                print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))


if __name__ == '__main__':
    pt = plateNet_train(image_size=(31, 94),
                        num_epoch=5,
                        batch_size=64,
                        learning_rate=0.001,
                        num_digit=8,
                        num_classes=65,
                        filewriter_path=os.path.join(PROJECT_PATH, "tmp/platenet/tensorboard"),
                        checkpoint_path=os.path.join(PROJECT_PATH, "tmp/platenet/process2_checkpoints"),
                        relu_leakiness=0.1,
                        is_restore=True,
                        restore_path=os.path.join(PROJECT_PATH, 'tmp/platenet/smooth_checkpoints'),
                        device_id='3'
                        )
    # pt.train()
    x, y = pt.gen_data()
    print(x, y)