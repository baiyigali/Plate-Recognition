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
    def __init__(self, image_size, num_epoch, batch_size, learning_rate, weight_decay,
                 num_digit, num_classes, train_records, valid_records, filewriter_path, checkpoint_path,
                 relu_leakiness=0.1, is_restore=True, restore_path=None, device_id='2'):
        self.image_size = image_size
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.train_batches_per_epoch = int(300000 / self.batch_size)
        self.valid_batches_per_epoch = int(300 / self.batch_size)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_digit = num_digit
        self.num_classes = num_classes
        self.train_records = train_records
        self.valid_records = valid_records,
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
        self.model = plate_model.plate_Net(self.x, batch_size=self.batch_size, weight_decay=self.weight_decay, num_classes=self.num_classes)
        self.predict = self.model.network_model()
        self.accuracy = self.model.accuracy(self.predict, self.y)
        self.loss = self.model.loss(self.predict, self.y)
        self.train_op = self.optimize_model()
        self.merge_summary = tf.summary.merge_all()

    def optimize_model(self):
        self.variable_list = [v for v in tf.trainable_variables()]
        # print(self.variable_list)
        gradients = tf.gradients(self.loss, self.variable_list)
        gradients = list(zip(gradients, self.variable_list))
        with tf.variable_scope('optimize'):
            optimizer = tf.train.MomentumOptimizer(self.learning_rate, momentum=0.9)
            train_op = optimizer.apply_gradients(gradients)
        return train_op

    def read_and_decode(self, filenames):
        """ Return tensor to read from TFRecord """
        directory = filenames
        files = tf.train.match_filenames_once(directory)
        filename_queue = tf.train.string_input_producer(files, shuffle=True)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example,
                                           features={
                                               'labels': tf.FixedLenFeature([], tf.string),
                                               'image_raw': tf.FixedLenFeature([], tf.string),
                                           })
        # You can do more image distortion here for training dat
        image = tf.decode_raw(features['image_raw'], tf.uint8)
        image = tf.reshape(image, [30, 120, 3])

        image = tf.cast(image, tf.float64)
        image = tf.multiply(image, [127.5, 127.5, 127.5])
        image = tf.random_crop(image, [30 - np.random.randint(0, 5), 120 - np.random.randint(10), 3])
        # image = tf.image.random_brightness(image, max_delta=32. / 255.)  # 亮度
        # image = tf.image.random_saturation(image, lower=0.5, upper=1.5)  # 饱和度
        # image = tf.image.random_hue(image, max_delta=0.2)  # 色相
        # image = tf.image.random_contrast(image, lower=0.5, upper=1.5)  # 对比度
        # image = tf.image.per_image_standardization(image)
        image = tf.image.resize_images(image, (31, 94))

        label = tf.decode_raw(features['labels'], tf.int64)
        label = tf.reshape(label, [8, ])
        label = tf.one_hot(label, self.num_classes, 1, 0)
        label = tf.expand_dims(label, axis=0)
        label = tf.cast(label, tf.float64)

        images, labels = tf.train.shuffle_batch([image, label], batch_size=self.batch_size,
                                                capacity=2000,
                                                min_after_dequeue=1000)
        return images, labels

    def train(self):
        self.writer = tf.summary.FileWriter(self.filewriter_path)
        self.saver = tf.train.Saver(max_to_keep=10)

        train_images, train_labels = self.read_and_decode(self.train_records)
        valid_images, valid_labels = self.read_and_decode(self.valid_records)
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
            init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
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
                    image_batch, label_batch = sess.run([train_images, train_labels])

                    feed_dict = {self.x: image_batch, self.y: label_batch}

                    sess.run(self.train_op, feed_dict=feed_dict)
                    # print("Iter {}/{}".format(step * self.batch_size, self.train_batches_per_epoch * self.batch_size))

                    if step % self.display_step == 0:
                        feed_dict = {self.x: image_batch, self.y: label_batch}
                        sess.run(self.train_op, feed_dict=feed_dict)
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
                    image_batch, label_batch = sess.run([valid_images, valid_labels])
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

            coordinator.request_stop()
            coordinator.join(threads)


if __name__ == '__main__':
    pt = plateNet_train(image_size=(31, 94),
                        num_epoch=20,
                        batch_size=200,
                        learning_rate=0.001,
                        weight_decay=0.066,
                        num_digit=8,
                        num_classes=65,
                        train_records="../../plate_dataset_new/based_real2/train.tfrecords-plate-*",
                        valid_records="../../plate_dataset_new/based_real2/valid.tfrecords-plate-*",
                        # train_records="../../plate_dataset_new/records_resized_15/train.tfrecords-plate-*",
                        # valid_records="../../plate_dataset_new/records_resized_15/valid.tfrecords-plate-*",
                        filewriter_path=os.path.join(PROJECT_PATH, "tmp/platenet/tensorboard"),
                        checkpoint_path=os.path.join(PROJECT_PATH, "tmp/platenet/process_records_checkpoints"),
                        relu_leakiness=0.1,
                        is_restore=True,
                        restore_path=os.path.join(PROJECT_PATH, 'tmp/platenet/base_line_checkpoint'),
                        device_id='0'
                        )
    pt.train()

    # train_images, train_labels = pt.read_and_decode("../../plate_dataset_new/crop_plate_record/train.tfrecords-plate-*")
    # print(train_images,train_labels)
    # with tf.Session() as sess:
    #     init = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
    #     sess.run(init)
    #     coordinator = tf.train.Coordinator()
    #     threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)
    #     for i in range(10):
    #         image, label = sess.run([train_images, train_labels])
    #         print(image.shape, label.shape)
