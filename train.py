# encoding=utf-8

import os
import sys
sys.path.append("..")
sys.path.append("../..")
import time
import numpy as np
import tensorflow as tf
from datetime import datetime
import resnet_model
from datagenerator import ImageDataGenerator
from utils import mkdirs
from sklearn.metrics import confusion_matrix


class ResNet_train():
    def __init__(self, image_size, num_epoch, batch_size, learning_rate,
                 weight_decay, num_digit, num_classes, train_file, valid_file, filewriter_path, checkpoint_path,
                 num_residual_units, relu_leakiness=0.1, is_bottleneck=True, is_restore=True, device_id='2'):
        self.image_size = image_size
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_digit = num_digit
        self.num_classes = num_classes
        self.display_step = 20
        self.train_file = train_file
        self.valid_file = valid_file
        self.filewriter_path = filewriter_path
        mkdirs(self.filewriter_path)
        self.checkpoint_path = checkpoint_path
        mkdirs(self.checkpoint_path)
        self.num_residual_units = num_residual_units
        self.relu_leakiness = relu_leakiness
        self.is_bottlneck = is_bottleneck
        if is_restore:
            ckpt = tf.train.get_checkpoint_state(self.checkpoint_path)
            self.restore_checkpoint = ckpt.model_checkpoint_path
        else:
            self.restore_checkpoint = ''

        os.environ['CUDA_VISIBLE_DEVICES'] = device_id

    def init_model(self):
        self.x = tf.placeholder(tf.float32, [None, self.image_size[0], self.image_size[1], 3], name='input')
        self.y = tf.placeholder(tf.float32, [None, self.num_digit, self.num_classes])
        hps = resnet_model.HParams(batch_size=self.batch_size,
                                   num_classes=self.num_classes,
                                   num_digit=self.num_digit,
                                   num_residual_units=self.num_residual_units,
                                   use_bottleneck=self.is_bottlneck,
                                   relu_leakiness=self.relu_leakiness,
                                   weight_decay_rate=self.weight_decay)
        self.model = resnet_model.ResNet(hps, self.x, self.y)
        self.output = self.model.out
        # predict = self.model.out
        # self.output = tf.nn.softmax(predict, name='output')

    def set_cost_function(self):
        self.var_list = [v for v in tf.trainable_variables()]
        self.output = tf.reshape(self.output, [self.batch_size, self.num_digit, self.num_classes])
        with tf.name_scope("cross_ent"):
            class_delta = self.output - self.y
            self.cost = tf.reduce_mean(tf.reduce_sum(tf.square(class_delta), axis=2))
            # self.cost = tf.reduce_mean(
            #     tf.nn.softmax_cross_entropy_with_logits(logits=self.output, labels=self.y))
            # self.cost += self.model._decay()

        # Train op
        with tf.name_scope("train"):
            # Get gradients of all trainable variables
            gradients = tf.gradients(self.cost, self.var_list)
            self.gradients = list(zip(gradients, self.var_list))

            # Create optimizer and apply gradient descent to the trainable variables
            # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            optimizer = tf.train.MomentumOptimizer(self.weight_decay, 0.9)
            self.train_op = optimizer.apply_gradients(grads_and_vars=self.gradients)

    def get_accuray(self):
        # Evaluation op: Accuracy of the model
        with tf.name_scope("accuracy"):
            prediction = tf.equal(tf.argmax(self.output, 2), tf.argmax(self.y, 2))
            self.accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

    def set_summary(self):
        for gradient, var in self.gradients:
            tf.summary.histogram(var.name + '/gradient', gradient)

        for var in self.var_list:
            tf.summary.histogram(var.name, var)

        # Add the loss to summary
        tf.summary.scalar('cross_entropy', self.cost)
        # Add the accuracy to the summary
        tf.summary.scalar('accuracy', self.accuracy)

        # Merge all summaries together
        self.merged_summary = tf.summary.merge_all()

        # Initialize the FileWriter
        self.writer = tf.summary.FileWriter(self.filewriter_path)

    def init_saver(self):
        self.saver = tf.train.Saver()

    def create_date(self):
        return ImageDataGenerator(self.train_file, scale_size=self.image_size, num_digit=self.num_digit,
                                  num_classes=self.num_classes), \
               ImageDataGenerator(self.valid_file, scale_size=self.image_size, num_digit=self.num_digit,
                                  num_classes=self.num_classes)

    def fit(self):
        self.init_model()
        self.set_cost_function()
        self.get_accuray()
        self.set_summary()
        self.init_saver()

        train_generator, val_generator = self.create_date()

        # Get the number of training/validation steps per epoch
        train_batches_per_epoch = np.floor(train_generator.data_size / self.batch_size).astype(np.int16)
        val_batches_per_epoch = np.floor(val_generator.data_size / self.batch_size).astype(np.int16)

        # Start Tensorflow session
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
            sess.run(tf.global_variables_initializer())

            self.writer.add_graph(sess.graph)

            # Load the pretrained weights into the non-trainable layer
            # if restore_checkponit is '' use ariginal weights, else use checkponit
            if not self.restore_checkpoint == '':
                self.saver.restore(sess, self.restore_checkpoint)

            print("{} Start training...".format(datetime.now()))
            print("{} Open Tensorboard :tensorboard --logdir {} --host localhost --port 6006".format(datetime.now(),
                                                                                                     self.filewriter_path))
            # Loop over number of epochs
            for epoch in range(self.num_epoch):

                print("Epoch number: {}/{}".format(epoch + 1, self.num_epoch))

                step = 1

                while step < train_batches_per_epoch:
                    # Get a batch of images and labels
                    batch_xs, batch_ys = train_generator.next_batch(self.batch_size)
                    # And run the training op
                    feed_dict = {self.x: batch_xs, self.y: batch_ys}
                    sess.run(self.train_op, feed_dict=feed_dict)

                    # Generate summary with the current batch of data and write to file
                    if step % self.display_step == 0:
                        loss, acc, s = sess.run([self.cost, self.accuracy, self.merged_summary], feed_dict=feed_dict)
                        self.writer.add_summary(s, epoch * train_batches_per_epoch + step)
                        print("Iter {}/{}, training mini-batch loss = {:.5f}, training accuracy = {:.5f}".format(
                            step * self.batch_size, train_batches_per_epoch * self.batch_size, loss, acc))
                        val_generator.reset_pointer()

                    step += 1

                # Validate the model on the entire validation set
                print("{} Start validation".format(datetime.now()))
                v_loss = 0.
                v_acc = 0.
                count = 0
                t1 = time.time()
                y_predict = np.zeros((self.batch_size, self.num_classes))
                # conf_matrix = np.ndarray((num_classes, num_classes))
                print("valid batchs {}".format(val_batches_per_epoch))
                for i in range(val_batches_per_epoch):
                    batch_validx, batch_validy = val_generator.next_batch(self.batch_size)
                    valid_loss, valid_acc, valid_out = sess.run([self.cost, self.accuracy, self.output],
                                                                feed_dict={self.x: batch_validx, self.y: batch_validy})

                    v_loss += valid_loss
                    v_acc += valid_acc
                    count += 1

                    # y_true = np.argmax(batch_validy, 1)
                    # y_pre = np.argmax(valid_out, 1)
                    # for k in range(self.batch_size):
                    #     if not (y_pre[k] == 0 or y_pre[k] == 1):
                    #         y_pre[k] = 0
                    #
                    # if i == 0:
                    #     conf_matrix = confusion_matrix(y_true, y_pre)
                    # else:
                    #     conf_matrix += confusion_matrix(y_true, y_pre)
                    #     # print(i, conf_matrix)
                v_loss /= count
                v_acc /= count
                t2 = time.time() - t1
                print("Validation loss = {:.4f}, acc = {:.4f}".format(v_loss, v_acc))
                print("Test image {:.4f}ms per image".format(t2 * 1000 / (val_batches_per_epoch * self.batch_size)))
                # print(conf_matrix)

                # Reset the file pointer of the image data generator
                val_generator.reset_pointer()
                train_generator.reset_pointer()

                print("{} Saving checkpoint of model...".format(datetime.now()))

                # save checkpoint of the model
                checkpoint_name = os.path.join(self.checkpoint_path, 'epoch_' + str(epoch))
                self.saver.save(sess, checkpoint_name)

                print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))


rt = ResNet_train(image_size=(100, 30),
                  num_epoch=80,
                  batch_size=64,
                  learning_rate=0.01,
                  weight_decay=0.0002,
                  num_digit=8,
                  num_classes=82,
                  train_file="./path/train.txt",
                  valid_file="./path/valid.txt",
                  filewriter_path="./tmp/resnet13/tensorboard",
                  checkpoint_path="./tmp/resnet13/checkpoints",
                  num_residual_units=2,
                  relu_leakiness=0.1,
                  is_bottleneck=True,
                  is_restore=False,
                  device_id='0'
                  )

rt.fit()

rt = ResNet_train(image_size=(100, 30),
                  num_epoch=100,
                  batch_size=64,
                  learning_rate=0.001,
                  weight_decay=0.0002,
                  num_digit=8,
                  num_classes=82,
                  train_file="./path/train.txt",
                  valid_file="./path/valid.txt",
                  filewriter_path="./tmp/resnet13/tensorboard",
                  checkpoint_path="./tmp/resnet13/checkpoints",
                  num_residual_units=2,
                  relu_leakiness=0.1,
                  is_bottleneck=True,
                  is_restore=False,
                  device_id='0'
                  )

rt.fit()
