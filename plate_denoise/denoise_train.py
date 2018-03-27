import tensorflow as tf
import os
from plate_denoise import denoise_model
import time
from datetime import datetime
import numpy as np
from plate_denoise.datagenerator import ImageDataGenerator
import utils

class ae_train():
    def __init__(self, image_size, num_epochs, batch_sizes, learning_rate, train_file, filewriter_path, checkpoint_path,
                 is_restore, restore_path, device_id):
        self.image_size = image_size
        self.num_epoch = num_epochs
        self.batch_size = batch_sizes
        self.learing_rate = learning_rate
        self.train_file = train_file
        self.filewriter_path = filewriter_path
        utils.deldirs(self.filewriter_path)
        utils.mkdirs(self.filewriter_path)
        self.checkpoint_path = checkpoint_path
        if not os.path.exists(filewriter_path):
            os.makedirs(filewriter_path)
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        self.display_step = 20
        if is_restore and os.path.exists(restore_path):
            ckpt = tf.train.get_checkpoint_state(restore_path)
            self.restore_checkpoint = ckpt.model_checkpoint_path
        else:
            self.restore_checkpoint = None

        os.environ['CUDA_VISIBLE_DEVICES'] = device_id
        self.init_model()

    def init_model(self):
        self.shear_images = tf.placeholder(tf.float32, [None, self.image_size[0], self.image_size[1], 3], name='input')
        self.source_images = tf.placeholder(tf.float32, [None, self.image_size[0], self.image_size[1], 3])
        self.model = denoise_model.denoise_autoencoder(self.shear_images)
        self.predict = self.model.network_model()
        self.loss = self.model.loss(self.predict, self.source_images)
        self.train_op = self.optimize_model()
        self.merge_summary = tf.summary.merge_all()

    def optimize_model(self):
        self.variable_list = [v for v in tf.trainable_variables()]
        gradients = tf.gradients(self.loss, self.variable_list)
        gradients = list(zip(gradients, self.variable_list))
        with tf.variable_scope('optimize'):
            optimizer = tf.train.MomentumOptimizer(self.learing_rate, momentum=0.9)
            train_op = optimizer.apply_gradients(gradients)
        return train_op

    def create_data(self):
        return ImageDataGenerator(self.train_file, ('plate_process_image_with_shear', 'license'), (30, 100))

    def train(self):
        self.writer = tf.summary.FileWriter(self.filewriter_path)
        self.saver = tf.train.Saver()

        train_generator = self.create_data()
        # Get the number of training/validation steps per epoch
        train_batches_per_epoch = np.floor(train_generator.data_size / self.batch_size).astype(np.int16)
        # val_batches_per_epoch = np.floor(val_generator.data_size / self.batch_size).astype(np.int16)

        # Start Tensorflow session
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
            sess.run(tf.global_variables_initializer())

            self.writer.add_graph(sess.graph)

            # Load the pretrained weights into the non-trainable layer
            # if restore_checkponit is '' use ariginal weights, else use checkponit
            if not self.restore_checkpoint is None:
                self.saver.restore(sess, self.restore_checkpoint)

            print("{} Start training...".format(datetime.now()))
            print("{} Open Tensorboard:tensorboard --logdir {} --host localhost --port 6066".format(datetime.now(),
                                                                                                     self.filewriter_path))
            # Loop over number of epochs
            for epoch in range(self.num_epoch):

                print("Epoch number: {}/{}".format(epoch + 1, self.num_epoch))

                step = 1

                while step < train_batches_per_epoch:

                    batch_xs, batch_ys = train_generator.next_batch(self.batch_size)

                    feed_dict = {self.shear_images: batch_xs, self.source_images: batch_ys}

                    sess.run(self.train_op, feed_dict=feed_dict)

                    # Generate summary with the current batch of data and write to file
                    if step % self.display_step == 0:
                        loss, s = sess.run([self.loss, self.merge_summary], feed_dict=feed_dict)
                        self.writer.add_summary(s, epoch * train_batches_per_epoch + step)
                        print("Iter {}/{}, training mini-batch loss = {:.5f}".format(
                            step * self.batch_size, train_batches_per_epoch * self.batch_size, loss))

                    step += 1
                train_generator.reset_pointer()


                print("{} Saving checkpoint of model...".format(datetime.now()))
                # save checkpoint of the model
                checkpoint_name = os.path.join(self.checkpoint_path, str(epoch + 1))
                self.saver.save(sess, checkpoint_name)

                print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))

if __name__ == '__main__':
    # train_file = "./path/license/train.txt"
    # valid_file = "./path/license/valid.txt"

    # train_file = "./path/plate_process_image/train.txt"
    # valid_file = "./path/plate_process_image/valid.txt"
    #
    train_file = "../path/plate_process_image_with_shear/train.txt"
    # valid_file = "./path/plate_process_image_with_shear/valid.txt"
    #
    # train_file = "./path/palte_process_image_without_shape/train.txt"
    # valid_file = "./path/palte_process_image_without_shape/valid.txt"

    pt = ae_train(image_size=(30, 100),
                        num_epochs=200,
                        batch_sizes=64,
                        learning_rate=0.001,
                        train_file=train_file,
                        filewriter_path="../tmp/ae/tensorboard",
                        checkpoint_path="../tmp/ae/shear_checkpoints",
                        is_restore=False,
                        restore_path='../tmp/ae/smooth_checkpoints',
                        device_id='5'
                        )
    pt.train()
