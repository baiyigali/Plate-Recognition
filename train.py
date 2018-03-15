# encoding=utf-8

import os
import sys

sys.path.append("..")
sys.path.append("../..")
import time
import numpy as np
import tensorflow as tf
from datetime import datetime
import fcn_model
from datagenerator import ImageDataGenerator
import utils
from sklearn.metrics import confusion_matrix


class FCN_train():
    def __init__(self, image_size, num_epoch, batch_size, learning_rate,
                 num_digit, num_classes, train_file, valid_file, filewriter_path, checkpoint_path,
                 relu_leakiness=0.1, is_restore=True, device_id='2'):
        self.image_size = image_size
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_digit = num_digit
        self.num_classes = num_classes
        self.display_step = 20
        self.train_file = train_file
        self.valid_file = valid_file
        self.filewriter_path = filewriter_path
        utils.deldirs(self.filewriter_path)
        utils.mkdirs(self.filewriter_path)
        self.checkpoint_path = checkpoint_path
        utils.mkdirs(self.checkpoint_path)
        self.relu_leakiness = relu_leakiness
        if is_restore:
            ckpt = tf.train.get_checkpoint_state(self.checkpoint_path)
            self.restore_checkpoint = ckpt.model_checkpoint_path
        else:
            self.restore_checkpoint = ''

        os.environ['CUDA_VISIBLE_DEVICES'] = device_id

    def init_model(self):
        self.x = tf.placeholder(tf.float32, [None, self.image_size[0], self.image_size[1], 3], name='input')
        self.y = tf.placeholder(tf.float32, [None, self.num_digit, self.num_classes])

        self.model = fcn_model.FCN(self.x, self.num_classes)
        # self.output = self.model.out
        # predict = self.model.out
        # self.output = tf.nn.softmax(predict, name='output')

    def set_loss(self):

        with tf.variable_scope('loss0') as scope:
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.model.fc0, labels=self.y[:, 0, :])
            self.loss0 = tf.reduce_mean(cross_entropy, name='loss0')
            tf.summary.scalar(scope.name + '/loss0', self.loss0)

        with tf.variable_scope('loss1') as scope:
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.model.fc1, labels=self.y[:, 1, :])
            self.loss1 = tf.reduce_mean(cross_entropy, name='loss1')
            tf.summary.scalar(scope.name + '/loss1', self.loss1)

        with tf.variable_scope('loss2') as scope:
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.model.fc2, labels=self.y[:, 2, :])
            self.loss2 = tf.reduce_mean(cross_entropy, name='loss2')
            tf.summary.scalar(scope.name + '/loss2', self.loss2)

        with tf.variable_scope('loss3') as scope:
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.model.fc3, labels=self.y[:, 3, :])
            self.loss3 = tf.reduce_mean(cross_entropy, name='loss3')
            tf.summary.scalar(scope.name + '/loss3', self.loss3)

        with tf.variable_scope('loss4') as scope:
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.model.fc4, labels=self.y[:, 4, :])
            self.loss4 = tf.reduce_mean(cross_entropy, name='loss4')
            tf.summary.scalar(scope.name + '/loss4', self.loss4)

        with tf.variable_scope('loss5') as scope:
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.model.fc5, labels=self.y[:, 5, :])
            self.loss5 = tf.reduce_mean(cross_entropy, name='loss5')
            tf.summary.scalar(scope.name + '/loss5', self.loss5)

        with tf.variable_scope('loss6') as scope:
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.model.fc6, labels=self.y[:, 6, :])
            self.loss6 = tf.reduce_mean(cross_entropy, name='loss6')
            tf.summary.scalar(scope.name + '/loss6', self.loss6)

        with tf.variable_scope('loss7') as scope:
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.model.fc7, labels=self.y[:, 7, :])
            self.loss7 = tf.reduce_mean(cross_entropy, name='loss7')
            tf.summary.scalar(scope.name + '/loss7', self.loss7)

    def set_optimizer(self):
        self.var_list = [v for v in tf.trainable_variables()]

        with tf.name_scope('optimizer0'):
            optimizer0 = tf.train.AdamOptimizer(self.learning_rate)
            self.train_op0 = optimizer0.minimize(self.loss0)

        with tf.name_scope('optimizer1'):
            optimizer1 = tf.train.AdamOptimizer(self.learning_rate)
            self.train_op1 = optimizer1.minimize(self.loss1)

        with tf.name_scope('optimizer2'):
            optimizer2 = tf.train.AdamOptimizer(self.learning_rate)
            self.train_op2 = optimizer2.minimize(self.loss2)

        with tf.name_scope('optimizer0'):
            optimizer3 = tf.train.AdamOptimizer(self.learning_rate)
            self.train_op3 = optimizer3.minimize(self.loss3)

        with tf.name_scope('optimizer0'):
            optimizer4 = tf.train.AdamOptimizer(self.learning_rate)
            self.train_op4 = optimizer4.minimize(self.loss4)

        with tf.name_scope('optimizer0'):
            optimizer5 = tf.train.AdamOptimizer(self.learning_rate)
            self.train_op5 = optimizer5.minimize(self.loss5)

        with tf.name_scope('optimizer0'):
            optimizer6 = tf.train.AdamOptimizer(self.learning_rate)
            self.train_op6 = optimizer6.minimize(self.loss6)

        with tf.name_scope('optimizer0'):
            optimizer7 = tf.train.AdamOptimizer(self.learning_rate)
            self.train_op7 = optimizer7.minimize(self.loss7)

    # def set_cost_function(self):
    #     self.var_list = [v for v in tf.trainable_variables()]
    #
    #     with tf.name_scope("cross_ent"):
    #         self.cost = tf.reduce_mean(
    #             tf.nn.softmax_cross_entropy_with_logits(logits=self.output, labels=self.y))
    #         self.cost += self.model._decay()
    #
    #     # Train op
    #     with tf.name_scope("train"):
    #         # Get gradients of all trainable variables
    #         gradients = tf.gradients(self.cost, self.var_list)
    #         self.gradients = list(zip(gradients, self.var_list))
    #
    #         # Create optimizer and apply gradient descent to the trainable variables
    #         # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    #         optimizer = tf.train.MomentumOptimizer(self.weight_decay, 0.9)
    #         self.train_op = optimizer.apply_gradients(grads_and_vars=self.gradients)
    def get_accurcy(self):
        logits = tf.concat([self.model.fc0, self.model.fc1, self.model.fc2, self.model.fc3,
                            self.model.fc4, self.model.fc5, self.model.fc6, self.model.fc7], 0)
        labels = tf.concat([self.y[:, 0, :], self.y[:, 1, :], self.y[:, 2, :], self.y[:, 3, :],
                            self.y[:, 4, :], self.y[:, 5, :], self.y[:, 6, :], self.y[:, 7, :]], 0)

        with tf.variable_scope('accuracy') as scope:
            prediton = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
            self.accuracy = tf.reduce_mean(tf.cast(prediton, tf.float32))
            tf.summary.scalar(scope.name + '/accuracy', self.accuracy)

    # def get_accurcy(self):
    #     out = tf.reshape(self.output, (self.batch_size, self.num_digit, self.num_classes))
    #     y = tf.reshape(self.y, (self.batch_size, self.num_digit, self.num_classes))
    #     # Evaluation op: Accuracy of the model
    #     with tf.name_scope("accuracy"):
    #         prediction = tf.equal(tf.argmax(out, 2), tf.argmax(y, 2))
    #         self.accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

    def set_summary(self):
        # for gradient, var in self.gradients:
        #     tf.summary.histogram(var.name + '/gradient', gradient)

        for var in self.var_list:
            tf.summary.histogram(var.name, var)

        # Add the loss to summary
        # tf.summary.scalar('cross_entropy', self.cost)
        # Add the accuracy to the summary
        # tf.summary.scalar('accuracy', self.accuracy)

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
        self.set_loss()
        self.set_optimizer()
        self.get_accurcy()
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

                print("Epoch number: {}/{}".format(epoch, self.num_epoch))

                step = 1
                while step < train_batches_per_epoch:
                    # Get a batch of images and labels
                    batch_xs, batch_ys = train_generator.next_batch(self.batch_size)
                    # And run the training op
                    feed_dict = {self.x: batch_xs, self.y: batch_ys}
                    sess.run([self.train_op0, self.train_op1, self.train_op2, self.train_op3,
                              self.train_op4, self.train_op5, self.train_op6, self.train_op7],
                             feed_dict=feed_dict)

                    # Generate summary with the current batch of data and write to file
                    if step % self.display_step == 0:
                        loss0, loss1, loss2, loss3, loss4, loss5, loss6, loss7, acc, s = \
                            sess.run([self.loss0, self.loss1, self.loss2, self.loss3,
                                      self.loss4, self.loss5, self.loss6, self.loss7,
                                      self.accuracy, self.merged_summary], feed_dict=feed_dict)
                        self.writer.add_summary(s, epoch * train_batches_per_epoch + step)
                        print(
                            "Iter {}/{}, training mini-batch loss0 = {:.4f}, loss1 = {:.4f}, "
                            "loss2 = {:.4f}, loss3 = {:.4f}, loss4 = {:.4f}, loss5 = {:.4f}, "
                            "loss6 = {:.4f},loss7 = {:.4f}, acc = {:.4f}".format(
                                step * self.batch_size, train_batches_per_epoch * self.batch_size,
                                loss0, loss1, loss2, loss3, loss4, loss5, loss6, loss7, acc))

                    step += 1

                # Validate the model on the entire validation set
                print("{} Start validation".format(datetime.now()))
                v_loss0 = 0.
                v_loss1 = 0.
                v_loss2 = 0.
                v_loss3 = 0.
                v_loss4 = 0.
                v_loss5 = 0.
                v_loss6 = 0.
                v_loss7 = 0.
                v_acc = 0.
                count = 0
                t1 = time.time()
                for i in range(val_batches_per_epoch):
                    batch_validx, batch_validy = val_generator.next_batch(self.batch_size)
                    valid_loss0, valid_loss1, valid_loss2, valid_loss3, valid_loss4, \
                    valid_loss5, valid_loss6, valid_loss7, valid_acc = \
                        sess.run([self.loss0, self.loss1, self.loss2, self.loss3,
                                  self.loss4, self.loss5, self.loss6, self.loss7,
                                  self.accuracy], feed_dict={self.x: batch_validx, self.y: batch_validy})

                    v_loss0 += valid_loss0
                    v_loss1 += valid_loss1
                    v_loss2 += valid_loss2
                    v_loss3 += valid_loss3
                    v_loss4 += valid_loss4
                    v_loss5 += valid_loss5
                    v_loss6 += valid_loss6
                    v_loss7 += valid_loss7
                    v_acc += valid_acc
                    count += 1

                v_loss0 /= count
                v_loss1 /= count
                v_loss2 /= count
                v_loss3 /= count
                v_loss4 /= count
                v_loss5 /= count
                v_loss6 /= count
                v_loss7 /= count
                v_acc /= count
                t2 = time.time() - t1
                print(
                    "Validation loss0 = {:.4f}, loss1 = {:.4f}, loss2 = {:.4f}, loss3 = {:.4f}, loss4 = {:.4f},"
                    "loss5 = {:.4f}, loss6 = {:.4f},loss7 = {:.4f}, acc = {:.4f}".format(
                        v_loss0, v_loss1, v_loss2, v_loss3, v_loss4, v_loss5, v_loss6, v_loss7, v_acc))
                print("Test image {:.4f}ms per image".format(t2 * 1000 / (val_batches_per_epoch * self.batch_size)))

                # Reset the file pointer of the image data generator
                val_generator.reset_pointer()
                train_generator.reset_pointer()

                print("{} Saving checkpoint of model...".format(datetime.now()))

                # save checkpoint of the model
                checkpoint_name = os.path.join(self.checkpoint_path, 'epoch_' + str(epoch))
                self.saver.save(sess, checkpoint_name)

                print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))


rt = FCN_train(image_size=(100, 30),
               num_epoch=800,
               batch_size=64,
               learning_rate=0.01,
               num_digit=8,
               num_classes=34,  # max num classes
               train_file="./path/train.txt",
               valid_file="./path/valid.txt",
               filewriter_path="./tmp/fcn4/tensorboard",
               checkpoint_path="./tmp/fcn4/checkpoints",
               relu_leakiness=0.1,
               is_restore=False,
               device_id='0,1'
               )

rt.fit()
