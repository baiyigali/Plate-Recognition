import os, cv2
import numpy as np
import tensorflow as tf
from data_process.label import class_dict

def str2serial(plate_name):
    serial = []
    for i, p in enumerate(plate_name):
        # print(class_dict[p])
        serial.append(class_dict[p])
    return serial

def write_records(file_path, record_folder, record_scope, record_batch_size=1000):
    if not os.path.exists(record_folder):
        os.makedirs(record_folder)
    record_file_num = 0
    # tfrecords_writer = tf.python_io.TFRecordWriter(record_scope)
    with open(file_path) as file:
        lines = file.readlines()
        for i, l in enumerate(lines):
            if i % record_batch_size == 0:
                record_file_name = (os.path.join(record_folder, record_scope + '%.3d' % (record_file_num)))
                tfrecords_writer = tf.python_io.TFRecordWriter(record_file_name)
                record_file_num += 1
            print(i, l, record_file_name)
            items = l.split()
            image_path = items[0]
            image_data = cv2.imread(image_path)
            print(image_data.shape)
            image_label = items[1]
            image_raw = image_data.tobytes()
            image_label = np.array(str2serial(image_label)).astype(np.int64)
            print(image_label, image_label.shape)
            label = image_label.tobytes()
            print(type(image_data), type(image_data[0][0][0]), image_data.shape, type(image_label[1]))
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'labels': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label])),
                        'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw]))
                    }
                )
            )
            tfrecords_writer.write(example.SerializeToString())
        tfrecords_writer.close()

if __name__ == "__main__":
    file_path = '/home1/fsb/project/LPR/Plate-Recognition/path/crop_image/valid.txt'
    record_folder = '/home1/fsb/project/LPR/plate_dataset_new/crop_plate_record'
    record_scope = 'valid.tfrecords-plate-'
    write_records(file_path, record_folder, record_scope)
