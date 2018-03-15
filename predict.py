import os
import sys

sys.path.append("..")
sys.path.append("../..")
import time
import numpy as np
import tensorflow as tf
import cv2


class plate_predict():
    def __init__(self, checkpoint_path, device_id='0'):
        self.num_classes = 2
        self.image_size = 64
        self.mean = np.array([127.5, 127.5, 127.5])
        self.checkpoint_path = checkpoint_path
        os.environ['CUDA_VISIBLE_DEVICES'] = device_id

    def get_data(self, path):
        img = cv2.imread(path).astype('float64')
        img -= self.mean
        img = np.resize(img, (1, self.image_size, self.image_size, 3))
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
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_path)
        with tf.Session() as sess:
            new_saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
            new_saver.restore(sess, ckpt.model_checkpoint_path)
            tf.get_default_graph().as_graph_def()
            x = sess.graph.get_tensor_by_name('input:0')
            y = sess.graph.get_tensor_by_name('fcn/output:0')

            result = sess.run(y, feed_dict={x: img})
            label = np.argmax(result, 1)
            # print("predict label {}, confidence {}%".format(label[0], result[0][label[0]] * 100))
            predict = label[0]
            confidence = result[0][label[0]]
            result = (predict, confidence)
        return result


if __name__ == '__main__':
    path = "./plate_image/1000.png"
    checkpoint_path = "./tmp/platenet/smooth_checkpoints"
    platenet_pre = plate_predict(checkpoint_path=checkpoint_path, device_id='0')
    # result = plate_predict.predict(path)
    # print(result)
    platenet_pre.read_var_name()
