# encoding=utf-8
import os, cv2
import numpy as np
import concurrent.futures
from PIL import Image
import data_process.label as data_label
from data_process import plate_process
import tensorflow as tf

CLASS_DICT = data_label.class_dict
CLASS_CHAR = data_label.class_char
PROJECT_PATH = '/home1/fsb/project/LPR/Plate-Recognition'

bg_path = ['image/base_bg1.jpg', 'image/base_bg2.jpg', 'image/base_bg3.jpg',
           'image/base_bg4.jpg', 'image/base_bg5.jpg', 'image/base_bg6.jpg',
           'image/base_bg7.jpg', 'image/base_bg8.jpg', 'image/base_bg9.jpg',
           'image/base_bg10.jpg', 'image/base_bg11.jpg', 'image/base_bg12.jpg',
           'image/base_bg14.jpg', 'image/base_bg13.jpg']

X_coord = 20
Y_coord = 32
offset = 67
offset2 = 68
space = 58
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
        serial.append(np.random.randint(0, 30))
        serial.append(np.random.randint(41, 64))
        serial.append(np.random.choice([44, 46]))
        for i in range(5):
            serial.append(np.random.randint(31, 64))
        return serial

    # 给定序列 背景图片生成车牌
    def create_plate(self, serial):
        path = np.random.choice(bg_path, 1)[0]
        # print(os.path.abspath(path))
        background = cv2.imread(os.path.join(PROJECT_PATH, path))
        coords = np.zeros((8, 4))
        for i, char in enumerate(serial):
            (x, y), (w, h) = self._image_fusion(background,
                                                os.path.join(PROJECT_PATH, 'image', CLASS_CHAR[char] + '.jpg'),
                                                CHAR_COORD[i]['x'],
                                                CHAR_COORD[i]['y'])
            coords[i] = np.array([x, y, w, h])
        return background, coords

    def __gray_image(self, image):
        image = image * 0.7 + 100
        return image

    # 图像融合
    # Input
    # background: back ground image
    # logo_path: path of logo image
    def _image_fusion(self, background, logo_path, x, y):
        backgroung_rows, background_cols, _ = background.shape
        logo = cv2.imread(logo_path)
        rows, cols, _ = logo.shape
        # if x + rows > bound_rows or y + cols > bound_rows:
        #     print("融合图像边界溢出")
        roi = background[y:y + rows, x:x + cols]
        gray = cv2.cvtColor(logo, cv2.COLOR_RGB2GRAY)

        # 设定阈值 小于thresh的置0， 大于thresh的置maxval
        ret, mask = cv2.threshold(gray, thresh=120, maxval=255, type=cv2.THRESH_BINARY)
        # print(logo.shape, roi.shape, mask.shape)
        fusion = cv2.bitwise_and(roi, roi, mask=mask)
        # fusion = self.__gray_image(fusion)
        background[y:y + rows, x:x + cols] = fusion
        # cv2.rectangle(background, (x, y), (x + cols, y + rows), (0, 0, 255))
        # self.imshow(bg)
        return (x / background_cols, y / backgroung_rows), \
               (cols / background_cols, rows / backgroung_rows)

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
        return ''.join(name) + '.jpg'

    def draw_box(self, image, coords):
        rows, cols, _ = image.shape
        for coord in coords:
            x = int(coord[0] * cols)
            y = int(coord[1] * rows)
            w = int(coord[2] * cols)
            h = int(coord[3] * rows)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255))


if __name__ == "__main__":

    # process = plate_process.plate_process()
    # plate = plate()
    # # # # bg = cv2.imread('./image/base_bg1.jpg')
    # # # # logo = cv2.imread('./image/1.jpg')
    # # # # plate._image_fusion(bg, logo, 0, 0)
    # # #
    # serial = plate.create_random_serial()
    # # # print(serial)
    # # # serial = [0, 41, 44, 39, 32, 37, 34, 39]
    # plate_image, coords = plate.create_plate(serial)
    # plate_image, coords = process.process_pic_with_shape_change(plate_image, coords)
    # # print(coords)
    # plate.draw_box(plate_image, coords)
    # plate.imshow(plate_image)

    # ====batch create plate===
    save_folder = '../../plate_dataset_new/records'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    plate = plate()
    process = plate_process.plate_process()

    serial_list = []
    for i in range(20000):
        serial_list.append(plate.create_random_serial())


    def main(serial):
        plate_image, coords = plate.create_plate(serial)
        plate_image, coords = process.process_pic_with_shape_change(plate_image, coords)
        # name = plate.serial2str(serial)
        # image_path = os.path.join(save_folder, name)
        labels = np.array(serial)
        info_dict = {
            'coords': coords,
            # 'image_path': image_path,
            'labels': labels,
        }
        return plate_image, info_dict
        # plate.imshow(plate_image)


    record_file_num = 0
    train_file_num = 0
    valid_file_num = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=80) as executor:
        for i, (image, info_dict) in enumerate(executor.map(main, serial_list)):
            if i % 1000 == 0:
                if i < 17000:
                    file_name = ('train.tfrecords-plate-%.4d' % (record_file_num))
                    writer = tf.python_io.TFRecordWriter(os.path.join(save_folder, file_name))
                else:
                    file_name = ('valid.tfrecords-plate-%.4d' % (record_file_num))
                    writer = tf.python_io.TFRecordWriter(os.path.join(save_folder, file_name))
                record_file_num += 1
            image_raw = image.tobytes()
            coords = info_dict['coords']
            labels = info_dict['labels']
            coords_raw = coords.tobytes()
            labels_raw = labels.tobytes()
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw])),
                        'coords': tf.train.Feature(bytes_list=tf.train.BytesList(value=[coords_raw])),
                        'labels': tf.train.Feature(bytes_list=tf.train.BytesList(value=[labels_raw]))
                    }
                )
            )
            # print(type(image), type(image[0][0][0]), image.shape, type(coords[1][0]))
            print(i)
            writer.write(example.SerializeToString())
        writer.close()
