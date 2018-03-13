# encoding=utf-8
import sys
sys.path.append('..')
import cv2
import os
import random
import itertools
import numpy as np
import shutil

NUMBER_PATH = ['../image/0（合并）.png', '../image/1（合并）.png', '../image/2（合并）.png', '../image/3（合并）.png',
               '../image/4（合并）.png', '../image/5（合并）.png', '../image/6（合并）.png', '../image/7（合并）.png',
               '../image/8（合并）.png', '../image/9（合并）.png', ]

LETTER_PATH = [
    '../image/A（合并）.png', '../image/B（合并）.png', '../image/C（合并）.png',
    '../image/D（合并）.png', '../image/E（合并）.png', '../image/F（合并）.png',
    '../image/G（合并）.png', '../image/H（合并）.png', '../image/J（合并）.png',
    '../image/K（合并）.png', '../image/L（合并）.png', '../image/M（合并）.png',
    '../image/N（合并）.png', '../image/P（合并）.png', '../image/Q（合并）.png',
    '../image/R（合并）.png', '../image/S（合并）.png', '../image/T（合并）.png',
    '../image/U（合并）.png', '../image/V（合并）.png', '../image/W（合并）.png',
    '../image/X（合并）.png', '../image/Y（合并）.png', '../image/Z（合并）.png',
]

# [0-9]/[A-Z]:24个排除IO
LETTER_DICT = {
    0: '../image/0（合并）.png', 1: '../image/1（合并）.png', 2: '../image/2（合并）.png',
    3: '../image/3（合并）.png', 4: '../image/4（合并）.png', 5: '../image/5（合并）.png',
    6: '../image/6（合并）.png', 7: '../image/7（合并）.png', 8: '../image/8（合并）.png',
    9: '../image/9（合并）.png',
    'A': '../image/A（合并）.png', 'B': '../image/B（合并）.png', 'C': '../image/C（合并）.png',
    'D': '../image/D（合并）.png', 'E': '../image/E（合并）.png', 'F': '../image/F（合并）.png',
    'G': '../image/G（合并）.png', 'H': '../image/H（合并）.png', 'J': '../image/J（合并）.png',
    'K': '../image/K（合并）.png', 'L': '../image/L（合并）.png', 'M': '../image/M（合并）.png',
    'N': '../image/N（合并）.png', 'P': '../image/P（合并）.png', 'Q': '../image/Q（合并）.png',
    'R': '../image/R（合并）.png', 'S': '../image/S（合并）.png', 'T': '../image/T（合并）.png',
    'U': '../image/U（合并）.png', 'V': '../image/V（合并）.png', 'W': '../image/W（合并）.png',
    'X': '../image/X（合并）.png', 'Y': '../image/Y（合并）.png', 'Z': '../image/Z（合并）.png',
}

# 31个 排除港澳台
PROVINCE_PATH = [
    '../image/湘（合并）.png', '../image/津（合并）.png', '../image/鄂（合并）.png', '../image/渝（合并）.png',
    '../image/冀（合并）.png', '../image/鲁（合并）.png', '../image/辽（合并）.png', '../image/浙（合并）.png',
    '../image/吉（合并）.png', '../image/黑（合并）.png', '../image/新（合并）.png', '../image/云（合并）.png',
    '../image/琼（合并）.png', '../image/青（合并）.png', '../image/贵（合并）.png', '../image/蒙（合并）.png',
    '../image/宁（合并）.png', '../image/甘（合并）.png', '../image/闽（合并）.png', '../image/皖（合并）.png',
    '../image/苏（合并）.png', '../image/粤（合并）.png', '../image/豫（合并）.png', '../image/川（合并）.png',
    '../image/藏（合并）.png', '../image/陕（合并）.png', '../image/桂（合并）.png',
    '../image/京（合并）.png', '../image/沪（合并）.png', '../image/赣（合并）.png', '../image/晋（合并）.png'
]

# 31个 排除港澳台
PROVINCE_NAME = [
    '湘', '津', '鄂', '渝', '冀',
    '鲁', '辽', '浙', '吉', '黑',
    '新', '云', '琼', '青', '贵',
    '蒙', '宁', '甘', '闽', '皖',
    '苏', '粤', '豫', '川', '藏',
    '陕', '桂', '京', '沪', '赣',
    '晋'
]

TYPE = ['D', 'F']

LETTER = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G',
    'H', 'J', 'K', 'L', 'M', 'N', 'P',
    'Q', 'R', 'S', 'T', 'U', 'V', 'W',
    'X', 'Y', 'Z']

start_x = 4
start_y = 265 - 70
offset = 70
COORD = [(start_x, start_y), (start_x, start_y + 1 * offset), (start_x, start_y + 2 * offset),
         (start_x, start_y + 3 * offset), (start_x, start_y + 4 * offset), (start_x, start_y + 5 * offset)]


class create_license():
    def __init__(self, base_bg, save_path):
        self.base_bg = base_bg
        self.province = 28
        self.city = 'A'
        self.type = 'D'
        self.save_path = save_path
        # self.create_image()
        # self.test()
        # self.create_random_number()
        self.create_random_number_with_letter(count=6, number_count=3)
        # self.set_province(bg_image)
        # self.set_city(bg_image)
        # self.set_type(bg_image)
        # self.create_license(bg_image, (4, 'H', 2, 7, 8))
        # self.test()
        # self.start()
        pass

    def test(self):
        bg_image = cv2.imread(self.base_bg)
        self.set_province(bg_image, 0)
        self.set_city(bg_image, 'A')
        self.set_type(bg_image, self.type)
        self.create_license(bg_image, (1, 2, 3, 4, 5))
        name = self.get_name(PROVINCE_NAME[0], 'A', self.type, (1, 2, 3, 4, 5))
        self.save_image(name, bg_image)

    # 生成随机的五位车牌
    # number_count:数字的位数，设置为5：全是数字没有字母, 设置为4：4位数+1位字母
    # letter_range:字母的范围(排除I和O, 车牌中没有I和O)，24：[A-Z]， 10:[A-J]
    def create_random_number_with_letter(self, count=5, number_count=5, letter_range=24):
        number_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        letter_list = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G',
            'H', 'J', 'K', 'L', 'M', 'N', 'P',
            'Q', 'R', 'S', 'T', 'U', 'V', 'W',
            'X', 'Y', 'Z']

        # 从数字list中选出number_count个数字，从字母list中选取
        # number_select_list = [c for c in itertools.combinations(number_list, number_count)]
        # letter_select_list = [c for c in itertools.combinations(letter_list, 5 - number_count)]

        number_select_list = self.combo(number_list, number_count)
        letter_select_list = self.combo(letter_list, count - number_count)

        for i in range(31):
                for j in range(17):
                    num = 0
                    for nsl in number_select_list:  # 120
                        for lsl in letter_select_list:  # 2024
                            if random.random() < 0.0006:  # 按照概率生成 否则太多了
                                group = nsl + lsl
                                random.shuffle(group)
                                # print(group)
                                bg_image = cv2.imread(self.base_bg)
                                self.set_province(bg_image, i)
                                self.set_city(bg_image, letter_list[j])
                                # self.set_type(bg_image, TYPE[k])
                                self.create_license(bg_image, group)
                                name = self.get_name(PROVINCE_NAME[i], letter_list[j], '', group)
                                self.save_image(self.save_path, name, bg_image)
                                num += 1
                                # print(i, j, group, num, end=' ')

    # 批量生成车牌
    def create_image(self, license_list):
        for i in range(31):
            for j in range(15):
                for k in range(2):
                    bg_image = cv2.imread(self.base_bg)
                    self.set_province(bg_image, i)
                    self.set_city(bg_image, LETTER[j])
                    # self.set_type(bg_image, TYPE[k])
                    self.create_license(bg_image, random.shuffle(license_list))

    # 生成单个车牌
    def create(self, license_list, base_bg, province, city, type):
        for ll in license_list:
            bg_image = cv2.imread(base_bg)
            self.set_province(bg_image, province)
            self.set_city(bg_image, city)
            self.set_type(bg_image, type)
            self.create_license(bg_image, ll)

    def set_province(self, bg_image, province=28):
        province_image = self.read_image(PROVINCE_PATH[province])
        coord = (4, 20)
        self.set_logo(bg_image, province_image, coord)

    def set_city(self, bg_image, city='A'):
        city_image = self.read_image(LETTER_DICT[city])
        coord = (4, 90)
        self.set_logo(bg_image, city_image, coord)

    def set_type(self, bg_image, type='D'):
        type_image = self.read_image(LETTER_DICT[type])
        coord = (4, 195)
        self.set_logo(bg_image, type_image, coord)

    # 合成汽车牌照
    def create_license(self, bg_image, number_list):

        if not bg_image.data:
            print('the background image {} with wrong path'.format(self.base_bg))
            return

        len = number_list.__len__()
        for i in range(0, len):
            number_image = self.read_image(LETTER_DICT[number_list[i]])
            # pose_letter = cv2.resize(pose_letter, (70, 177))
            bg_image = self.set_logo(bg_image, number_image, COORD[i])

            # cv2.namedWindow('chepai')
            # cv2.imshow('chepai', bg_image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

    def set_logo(self, bg_image, logo, coord):
        pose_letter_h, pose_letter_w, _ = logo.shape

        pose_coord = coord

        bg_image[
        int(pose_coord[0]): int(pose_coord[0] + pose_letter_h),
        int(pose_coord[1]): int(pose_coord[1] + pose_letter_w),
        :
        ] = logo[:, :, 0:3]
        return bg_image

    def get_name(self, province, city, type, number):
        name = str(province) + str(city) + str(type)
        for n in number:
            name = name + str(n)
        name += '.png'
        return name

    def read_image(self, path):
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)

    def save_image(self, folder, name, image):
        if not os.path.exists(folder):
            os.makedirs(folder)
        path = os.path.normpath(os.path.join(folder, name))
        cv2.imwrite(path, image)
        print(path)
        # ll = os.listdir(self.save_path)
        # os.rename(os.path.join(self.save_path, ll[0]), os.path.join(self.save_path, name))
        # self.movefile(os.path.join(self.save_path, name), os.path.join('../license2', name))

    def movefile(self, srcfile, dstfile):
        if not os.path.isfile(srcfile):
            print("%s not exist!" % (srcfile))
        else:
            fpath, fname = os.path.split(dstfile)  # 分离文件名和路径
            if not os.path.exists(fpath):
                os.makedirs(fpath)  # 创建路径
            shutil.move(srcfile, dstfile)  # 移动文件

    def combo(self, list, size):
        if size == 0 or not list:
            return [list[:0]]
        else:
            result = []
            for i in range(0, (len(list) - size) + 1):
                pick = list[i:i + 1]
                rest = list[i + 1:]
                for x in self.combo(rest, size - 1):
                    result.append(pick + x)
        return result


if __name__ == '__main__':
    base_bg = '../image/base_bg.png'
    license = create_license(base_bg=base_bg, save_path='../../dataset/license')
