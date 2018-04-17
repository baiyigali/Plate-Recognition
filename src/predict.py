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
        img = cv2.imread(path).astype('float32')
        img -= self.mean
        img = cv2.resize(img, (self.image_size[1], self.image_size[0]))
        img = img.reshape((1, self.image_size[0], self.image_size[1], 3))
        return img


    def read_var_name(self):
        from tensorflow.python import pywrap_tensorflow
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_path)
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
                new_saver.restore(sess, ckpt.model_checkpoint_path)
                graph = tf.get_default_graph()
                # keys = sess.graph.get_operations()
                # for key in keys:
                #     print(key)
                x = graph.get_tensor_by_name('input:0')
                y = graph.get_tensor_by_name('fcn/output/Reshape_1:0')
                # x = graph.get_operation_by_name('input').outputs[0]
                # y = graph.get_collection("predict_network")[0]
                # print(y)

                result = sess.run(y, feed_dict={x: img})
        return result


if __name__ == '__main__':
    path = "/home1/fsb/project/LPR/plate_dataset/license/鲁C6D81EL.png"
    # path = "/home1/fsb/project/LPR/plate_dataset/license/贵P59KUC7.png"
    # path = "/home1/fsb/project/LPR/plate_dataset/license/闽NM849UD.png"
    # path = "/home1/fsb/project/LPR/plate_dataset/license/吉DT18N5B.png"
    path = "/home1/fsb/project/LPR/plate_dataset/license/冀J8MYR75.png"
    # path = "/home1/fsb/project/LPR/plate_dataset/license/粤D0G7V5X.png"



    checkpoint_path = "./tmp/platenet/resized_checkpoints"
    platenet_pre = plate_predict(
        checkpoint_path=checkpoint_path,
        device_id='')
    real_list = ['a2.jpg', 'a3.jpg', 'a4.jpg', 'a6.jpg', 'a7.jpg', 'a8.jpg', 'a9.jpg', 'a10.jpg',
                 'a11.jpg', 'a12.jpg', 'a13.jpg', 'a14.jpg', 'a15.jpg', 'a16.jpg']
    for name in real_list:
        path = os.path.join("/home1/fsb/project/LPR/plate_dataset/real_image/", name)

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