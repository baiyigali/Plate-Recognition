# encoding=utf-8
import cv2, os
import numpy as np
from PIL import Image
import concurrent.futures

CLASS_DICT = {"京": 0, "沪": 1, "津": 2, "渝": 3, "冀": 4,
              "晋": 5, "蒙": 6, "辽": 7, "吉": 8, "黑": 9,
              "苏": 10, "浙": 11, "皖": 12, "闽": 13, "赣": 14,
              "鲁": 15, "豫": 16, "鄂": 17, "湘": 18, "粤": 19,
              "桂": 20, "琼": 21, "川": 22, "贵": 23, "云": 24,
              "藏": 25, "陕": 26, "甘": 27, "青": 28, "宁": 29,
              "新": 30, "0": 31, "1": 32, "2": 33, "3": 34,
              "4": 34, "5": 36, "6": 37, "7": 38, "8": 39, "9": 40,
              "A": 41, "B": 42, "C": 43, "D": 44, "E": 45,
              "F": 46, "G": 47, "H": 48, "J": 49, "K": 50,
              "L": 51, "M": 52, "N": 53, "P": 54, "Q": 55,
              "R": 56, "S": 57, "T": 58, "U": 59, "V": 60,
              "W": 61, "X": 62, "Y": 63, "Z": 64}

CLASS_CHAR = ["京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑",
              "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤",
              "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新",
              "0", "1", "2", "3", "4",
              "5", "6", "7", "8", "9",
              "A", "B", "C", "D", "E",
              "F", "G", "H", "J", "K",
              "L", "M", "N", "P", "Q",
              "R", "S", "T", "U", "V",
              "W", "X", "Y", "Z"
              ]

X_coord = 32
Y_coord = 20
offset = 67
offset2 = 68
space = 58
CHAR_COORD = [
    {'x': X_coord + 6, 'y': Y_coord},
    {'x': X_coord, 'y': Y_coord + offset * 1},
    {'x': X_coord, 'y': Y_coord + offset2 * 2 + space},
    {'x': X_coord, 'y': Y_coord + offset2 * 3 + space},
    {'x': X_coord, 'y': Y_coord + offset2 * 4 + space},
    {'x': X_coord, 'y': Y_coord + offset2 * 5 + space},
    {'x': X_coord, 'y': Y_coord + offset2 * 6 + space},
    {'x': X_coord, 'y': Y_coord + offset2 * 7 + space}
]


class plate():
    def __init__(self):
        pass

    # 随机生成一个8位长度的序列
    def create_random_serial(self):
        serial = []
        serial.append(np.random.randint(0, 30))
        serial.append(np.random.randint(41, 64))
        if np.random.random() > 0.5:
            serial.append(44)
        else:
            serial.append(46)
        for i in range(5):
            serial.append(np.random.randint(31, 64))
        return serial

    # 给定序列 背景图片生成车牌
    def create_plate(self, serial, bg_path):
        print(bg_path)
        background = cv2.imread(bg_path)
        for i, char in enumerate(serial):
            self._image_fusion(background, './' + CLASS_CHAR[char] + '.jpg', CHAR_COORD[i]['x'],
                               CHAR_COORD[i]['y'])
        return background

    # 图像融合
    # Input
    # background: back ground image
    # logo_path: path of logo image
    def _image_fusion(self, background, logo_path, x, y):
        bound_rows, blond_cols, _ = background.shape
        print(logo_path)
        logo = cv2.imread(logo_path)
        rows, cols, _ = logo.shape
        # if x + rows > bound_rows or y + cols > bound_rows:
        #     print("融合图像边界溢出")
        roi = background[x:x + rows, y:y + cols]
        gray = cv2.cvtColor(logo, cv2.COLOR_RGB2GRAY)

        # 设定阈值 小于thresh的置0， 大于thresh的置maxval
        ret, mask = cv2.threshold(gray, thresh=120, maxval=255, type=cv2.THRESH_BINARY)
        # print(logo.shape, roi.shape, mask.shape)
        fusion = cv2.bitwise_and(roi, roi, mask=mask)
        background[x:x + rows, y:y + cols] = fusion
        # self.imshow(bg)
        return background

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


if __name__ == "__main__":
    plate = plate()

    # bg = cv2.imread('./base_bg1.jpg')
    # logo = cv2.imread('./1.jpg')
    # plate._image_fusion(bg, logo, 0, 0)

    serial = plate.create_random_serial()
    print(serial)

    plate_image = plate.create_plate(serial, './base_bg2.jpg')
    plate.imshow(plate_image)

    # save_folder = '../../plate_dataset/license'
    # bg_path = ['../image/base_bg1.jpg', '../image/base_bg2.jpg', '../image/base_bg3.jpg',
    #            '../image/base_bg4.jpg', '../image/base_bg5.jpg', '../image/base_bg6.jpg']

    # plate = plate()
    # serial_list = []
    # for i in range(200000):
    #     serial_list.append(plate.create_random_serial())
    #
    #
    # def main(serial):
    #     plate_image = plate.create_plate(serial, np.random.choice(bg_path, 1)[0])
    #     name = plate.serial2str(serial)
    #     plate.save_image(image=plate_image, name=name, folder='../../plate_dataset_new/license')
    #     return name
    #     # plate.imshow(plate_image)
    #
    # with concurrent.futures.ThreadPoolExecutor(max_workers=80) as executor:
    #     for i, name in enumerate(executor.map(main, serial_list)):
    #         print(i, name)
