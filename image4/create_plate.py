# encoding=utf-8
import os, sys, cv2, time
import numpy as np
import tensorflow as tf
from PIL import Image
import data_process.label as data_label
from data_process import plate_process

CLASS_DICT = data_label.class_dict
CLASS_CHAR = data_label.class_char
PROJECT_PATH = '/home1/fsb/project/LPR/Plate-Recognition'

X_coord = 5
Y_coord = 0
offset = 17
offset2 = 17
space = 13
CHAR_COORD = [
    {'x': X_coord, 'y': Y_coord + 0},
    {'x': X_coord + offset * 1, 'y': Y_coord},
    {'x': X_coord + offset2 * 2 + space, 'y': Y_coord},
    {'x': X_coord + offset2 * 3 + space, 'y': Y_coord},
    {'x': X_coord + offset2 * 4 + space, 'y': Y_coord},
    {'x': X_coord + offset2 * 5 + space, 'y': Y_coord},
    {'x': X_coord + offset2 * 6 + space, 'y': Y_coord},
    {'x': X_coord + offset2 * 7 + space, 'y': Y_coord}
]


class plate():
    def __init__(self):
        pass

    # 随机生成一个8位长度的序列
    def create_random_serial(self):
        serial = []
        # serial.append(np.random.randint(0, 30))
        serial.append(19)  # yue
        number_list = [v for v in range(31, 41)]
        number_list += [42, 44, 46]
        # serial.append(np.random.choice([42, 44, 46]))
        # serial.append(np.random.choice([44, 46]))
        # for i in range(4):
        #     serial.append(np.random.randint(31, 40))
        # serial.append(np.random.choice([44, 46]))
        for i in range(7):
            serial.append(np.random.choice(number_list))
        return serial

    def _read_image(self, path):
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)

    def imshow(self, image):
        cv2.namedWindow("new")
        cv2.imshow("new", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save_image(self, image, name, folder='./'):
        if not os.path.exists(folder):
            os.makedirs(folder)
        path = os.path.join(folder, name)
        cv2.imwrite(path, image)
        print("save image at {}".format(path))

    def serial2str(self, serial):
        name = []
        for i, s in enumerate(serial):
            name.append(CLASS_CHAR[s])
        return ''.join(name)

    def create_plate(self, str):
        bg_list = os.listdir('./bg')
        bg_image = cv2.imread(os.path.join('./bg', np.random.choice(bg_list)))
        bg_image = cv2.resize(bg_image, (160, 40))
        for i, s in enumerate(str):
            logo_list = os.listdir(s)
            # print(logo_list)
            logo_image = cv2.imread(os.path.join(s, np.random.choice(logo_list)))
            logo_image = cv2.resize(logo_image, (17, 40))
            col, row, _ = logo_image.shape
            start_x = CHAR_COORD[i]['x']
            start_y = CHAR_COORD[i]['y']

            # print(i, start_x, start_y)
            bg_image[start_y:start_y + col, start_x:start_x + row, :] = logo_image

        return bg_image


if __name__ == "__main__":
    # plate = plate()
    # process = plate_process.plate_process()
    # serial = plate.create_random_serial()
    # str = plate.serial2str(serial)
    # print(serial, str)
    # plate_image = plate.create_plate(str)
    # print(plate_image.shape)
    # plate_image = process.process_pic_with_shape_change(plate_image)
    # print(plate_image.shape)
    # # plate.imshow(plate_image)
    # # plate_image = cv2.blur(plate_image, (3, 3))
    # plate.save_image(plate_image, 'new.jpg')
    #
    # sys.exit()

    # ====[START] create batch plate in record====
    import concurrent.futures

    save_folder = '../../plate_dataset_new/based_real2'
    create_num = 100000
    train_num = 80000
    t1 = time.time()
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    plate = plate()
    process = plate_process.plate_process()
    serial_list = []
    for i in range(create_num):
        serial_list.append(plate.create_random_serial())


    def main(serial):
        str = plate.serial2str(serial)
        plate_image = plate.create_plate(str)
        plate_image = process.process_pic_with_shape_change(plate_image)
        plate_image = cv2.resize(plate_image, (120, 30))
        # name = plate.serial2str(serial)
        # image_path = os.path.join(save_folder, name)
        labels = np.array(serial)
        # print(plate_image, labels)
        return plate_image, labels


    record_file_num = 0
    train_file_num = 0
    valid_file_num = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=60) as executor:
        for i, (image, labels) in enumerate(executor.map(main, serial_list)):
            if i % 1000 == 0:
                if i < train_num:
                    file_name = ('train.tfrecords-plate-%.4d' % (record_file_num))
                    writer = tf.python_io.TFRecordWriter(os.path.join(save_folder, file_name))
                else:
                    file_name = ('valid.tfrecords-plate-%.4d' % (record_file_num))
                    writer = tf.python_io.TFRecordWriter(os.path.join(save_folder, file_name))
                record_file_num += 1
            image_raw = image.tobytes()
            labels_raw = labels.tobytes()
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw])),
                        'labels': tf.train.Feature(bytes_list=tf.train.BytesList(value=[labels_raw]))
                    }
                )
            )
            # print(type(image), type(image[0][0][0]), image.shape, type(coords[1][0]))
            print(i)
            writer.write(example.SerializeToString())
        writer.close()

    print("cost time {}".format(time.time() - t1))
    # ====[END] create batch plate in record====
