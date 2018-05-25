# encoding=utf-8
import os
import sys

sys.path.append("..")
sys.path.append("../..")
import time
import numpy as np
import tensorflow as tf
import cv2
import data_process.label as data_label
import matplotlib.pyplot as plt


class plate_predict():
    def __init__(self, checkpoint_path, device_id):

        self.image_size = (31, 94)
        # self.image_size = (96, 33)
        self.mean = np.array([127.5, 127.5, 127.5])
        self.checkpoint_path = checkpoint_path
        os.environ['CUDA_VISIBLE_DEVICES'] = device_id
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    def get_data(self, path):
        img = cv2.imread(path).astype('float64')
        # img = cv2.blur(img, (5, 5))
        img -= self.mean
        img = cv2.resize(img, (self.image_size[1], self.image_size[0]))
        # img = tf.image.per_image_standardization(img)
        # img = img.reshape((1, self.image_size[0], self.image_size[1], 3))
        img = tf.expand_dims(img, 0)

        return img

    def read_var_name(self):
        from tensorflow.python import pywrap_tensorflow
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_path)
        print(ckpt.model_checkpoint_path)
        reader = pywrap_tensorflow.NewCheckpointReader(ckpt.model_checkpoint_path)
        var_to_shape_map = reader.get_variable_to_shape_map()
        for key in var_to_shape_map:
            print("tensor name: {}".format(key))

    def predict(self, path):
        img = self.get_data(path)
        # print(img.shape)
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_path)
        with tf.Session() as sess:
            new_saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
            if ckpt and ckpt.model_checkpoint_path:
                # print(ckpt, ckpt.model_checkpoint_path)
                print(ckpt.model_checkpoint_path)
                new_saver.restore(sess, ckpt.model_checkpoint_path)
                graph = tf.get_default_graph()
                # keys = sess.graph.get_operations()
                # for key in keys:
                #     print(key)
                x = graph.get_tensor_by_name('input:0')
                y = graph.get_tensor_by_name('fcn/output/Reshape_1:0')
                # is_train = graph.get_tensor_by_name('is_train:0')
                # x = graph.get_operation_by_name('input').outputs[0]
                # y = graph.get_collection("predict_network")[0]
                # print(y)
                image = sess.run(img)
                result = sess.run(y, feed_dict={x: image})
        return result


if __name__ == '__main__':

    checkpoint_path = "../tmp/platenet/process_records_checkpoints"
    # checkpoint_path = "../tmp/platenet/resized_checkpoints"
    platenet_pre = plate_predict(
        checkpoint_path=checkpoint_path,
        device_id='')

    # real_list = ['a2.jpg', 'a3.jpg', 'a4.jpg', 'a6.jpg', 'a7.jpg', 'a8.jpg', 'a9.jpg', 'a10.jpg',
                 # 'a11.jpg', 'a12.jpg', 'a13.jpg', 'a14.jpg', 'a15.jpg', 'a16.jpg']
    # real_list = ["/home1/fsb/project/LPR/plate_dataset/license/鲁C6D81EL.png"]
    # real_list = ['22_a.bmp', '280_a.bmp', '558_a.bmp', '581_a.bmp', '698_a.bmp', '603_a.bmp', '608_a.bmp', '729_a.bmp', '745_a.bmp']
    real_list = os.listdir("/home1/fsb/project/LPR/Plate-Recognition/test_plate")
    # real_list = ["/home1/fsb/project/LPR/Plate-Recognition/image2/new.jpg"]
    # real_list = ['22_a.jpg', '280_a.jpg', '558_a.jpg', '581_a.jpg', '698_a.jpg', '603_a.jpg', '608_a.jpg', '729_a.jpg', '745_a.jpg']
    real_list = ['test1.png', 'test2.png']
    for name in real_list:
        # path = os.path.join("/home1/fsb/project/LPR/plate_dataset/real_image/", name)
        path = os.path.join("/home1/fsb/project/LPR/Plate-Recognition/test_plate", name)
        # print(path)
        result = platenet_pre.predict(path)

        label = np.argmax(result, axis=3)
        plate_char = []
        for l in label[0, 0, :]:
            plate_char.append(data_label.class_char[l])
        predict = ''.join(plate_char)
        print(predict)
        plt.figure()
        # plt.title('plate predict')
        plt.imshow(plt.imread(path))
        plt.xlabel(predict)
        plt.show()
        plt.close()
