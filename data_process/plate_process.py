# encoding=utf-8
import sys

sys.path.append('..')
sys.path.append('../..')
import numpy as np
import cv2
import math
import random
import os


# 对图像进行错切变换、仿射变换、投影变换、运动模糊、高斯模糊、噪声处理
#
class plate_process():
    def __init__(self):
        pass

    # 错切变换
    def shear_mapping(self, image, angel=None, max_angel=10):
        shape = image.shape
        if angel is None:
            angel = random.random() * 10
        size_o = [shape[1], shape[0]]

        size = (shape[1] + int(shape[0] * math.cos((float(max_angel) / 180) * 3.14)), shape[0])

        interval = abs(int(math.sin((float(angel) / 180) * 3.14) * shape[0]))

        pts1 = np.float32([[0, 0], [0, size_o[1]], [size_o[0], 0], [size_o[0], size_o[1]]])
        if angel > 0:
            pts2 = np.float32([[interval, 0], [0, size[1]], [size[0], 0], [size[0] - interval, size_o[1]]])
        else:
            pts2 = np.float32([[0, 0], [interval, size[1]], [size[0] - interval, 0], [size[0], size_o[1]]])

        M = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(image, M, size)

        return dst

    def random_int(self, a=-10, b=10):
        return random.randint(a, b)

    # 放射变换
    def wrap_affine(self, image):
        rows, cols, _ = image.shape

        def up():
            # rd = random.randint(5, 20)
            srcpoint = np.float32([[0, 0], [0, rows], [cols, rows]])
            canvas = np.float32([[0 + self.random_int(), 0 + self.random_int()],
                                 [0 + self.random_int(), rows + self.random_int()],
                                 [cols + self.random_int(), rows + self.random_int()]])
            return srcpoint, canvas

        def right():
            rd = random.randint(5, 20)
            srcpoint = np.float32([[0, 0], [cols, 0], [cols, rows]])
            canvas = np.float32([[0 + self.random_int(), 0 + self.random_int()],
                                 [cols + self.random_int(), 0 + self.random_int()],
                                 [cols + self.random_int(), rows + self.random_int()]])
            return srcpoint, canvas

        def down():
            rd = random.randint(5, 20)
            srcpoint = np.float32([[cols, 0], [0, rows], [cols, rows]])
            canvas = np.float32([[cols + self.random_int(), 0 + self.random_int()],
                                 [0 + self.random_int(), rows + self.random_int()],
                                 [cols + self.random_int(), rows + self.random_int()]])
            return srcpoint, canvas

        def left():
            rd = random.randint(5, 20)
            srcpoint = np.float32([[0, 0], [0, rows], [cols, 0]])
            canvas = np.float32([[0 + self.random_int(), 0 + self.random_int()],
                                 [0 + self.random_int(), rows + self.random_int()],
                                 [cols + self.random_int(), 0 + self.random_int()]])
            return srcpoint, canvas

        switch = {0: up(),
                  1: right(),
                  2: down(),
                  3: left()}
        direction = random.randint(0, 3)
        srcpoint, canvas = switch[direction]

        M = cv2.getAffineTransform(srcpoint, canvas)

        dst = cv2.warpAffine(image, M, (cols, rows))
        return dst

    # 投影变换
    def projective_transform(self, image):
        rows, cols, _ = image.shape  # (185, 620)

        x_min = int(cols * 0.02)
        x_max = int(cols * 0.12)
        y_min = int(rows * 0.08)
        y_max = int(rows * 0.18)

        def up():
            rd = random.randint(x_min, x_max)
            srcpoint = np.float32([[0, 0], [0, rows], [cols, 0], [cols, rows]])
            canvas = np.float32([[0, 0], [0 + rd, rows], [cols, 0], [cols - rd, rows]])
            return srcpoint, canvas

        def upright():
            rd = random.randint(int(x_min / 2.), int(x_max / 2.))
            srcpoint = np.float32([[0, 0], [0, rows], [cols, 0], [cols, rows]])
            canvas = np.float32([[0, 0 + rd], [0 + int(rd / 2.), rows - int(rd / 2.)], [cols, 0], [cols - rd, rows]])
            return srcpoint, canvas

        def right():
            rd = random.randint(y_min, y_max)
            srcpoint = np.float32([[0, 0], [0, rows], [cols, 0], [cols, rows]])
            canvas = np.float32([[0, 0 + rd], [0, rows - rd], [cols, 0], [cols, rows]])
            return srcpoint, canvas

        def downright():
            rd = random.randint(int(x_min / 2.), int(x_max / 2.))
            srcpoint = np.float32([[0, 0], [0, rows], [cols, 0], [cols, rows]])
            canvas = np.float32([[0 + int(rd / 2.), 0 + int(rd / 2.)], [0, rows - rd], [cols - rd, 0], [cols, rows]])
            return srcpoint, canvas

        def down():
            rd = random.randint(x_min, x_max)
            srcpoint = np.float32([[0, 0], [0, rows], [cols, 0], [cols, rows]])
            canvas = np.float32([[0 + rd, 0], [0, rows], [cols - rd, 0], [cols, rows]])
            return srcpoint, canvas

        def downleft():
            rd = random.randint(int(x_min / 2.), int(x_max / 2.))
            srcpoint = np.float32([[0, 0], [0, rows], [cols, 0], [cols, rows]])
            canvas = np.float32([[0 + rd, 0], [0, rows], [cols - int(rd / 2.), 0 + int(rd / 2.)], [cols, rows - rd]])
            return srcpoint, canvas

        def left():
            rd = random.randint(y_min, y_max)
            srcpoint = np.float32([[0, 0], [0, rows], [cols, 0], [cols, rows]])
            canvas = np.float32([[0, 0], [0, rows], [cols, 0 + rd], [cols, rows - rd]])
            return srcpoint, canvas

        def upleft():
            rd = random.randint(int(x_min / 2.), int(x_max / 2.))
            srcpoint = np.float32([[0, 0], [0, rows], [cols, 0], [cols, rows]])
            canvas = np.float32([[0 + rd, 0], [0, rows], [cols - int(rd / 2.), 0 + int(rd / 2.)], [cols, rows - rd]])
            return srcpoint, canvas

        switch = {0: up(),
                  1: upright(),
                  2: right(),
                  3: downright(),
                  4: down(),
                  5: downleft(),
                  6: left(),
                  7: upleft()}
        direction = random.randint(0, 7)
        srcpoint, canvas = switch[direction]

        PerspectiveMatrix = cv2.getPerspectiveTransform(srcpoint, canvas)
        dst = cv2.warpPerspective(image, PerspectiveMatrix, (cols, rows))
        return dst

    # 运动模糊
    def motion_blur(self, image, degree=None, angle=None):
        image = np.array(image)
        if degree is None:
            degree = random.randint(5, 18)
        if angle is None:
            angle = random.randint(10, 180)

        # 这里生成任意角度的运动模糊kernel的矩阵， degree越大，模糊程度越高
        M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
        motion_blur_kernel = np.diag(np.ones(degree))
        motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))

        motion_blur_kernel = motion_blur_kernel / degree
        blurred = cv2.filter2D(image, -1, motion_blur_kernel)
        # convert to uint8
        cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
        dst = np.array(blurred, dtype=np.uint8)
        return dst

    # 对焦模糊、高斯模糊
    def gauss_blur(self, image, degree=12):
        if degree is None:
            degree = random.randint(2, 12)

        return cv2.blur(image, (degree, degree))

    # 噪声
    def gauss_noise(self, image, degree=None):
        row, col, ch = image.shape
        mean = 0
        if degree is None:
            degree = random.uniform(10, 100)

        sigma = degree ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss
        cv2.normalize(noisy, noisy, 0, 255, norm_type=cv2.NORM_MINMAX)
        dst = np.array(noisy, dtype=np.uint8)
        return dst

    # 图片高亮
    def change_light(self, image, degree=1.02):
        row, col, ch = image.shape
        print(image.shape)
        dst = image * degree
        return dst

    # 生成遮挡物
    def add_blocked(self, image):
        pass

    def read_image(self, path):
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)

    def save_image(self, image, name, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)
        path = os.path.join(folder, name)
        cv2.imwrite(path, image)
        print("save image at {}".format(path))

    def show_image(self, image):
        cv2.namedWindow('demo')
        cv2.imshow('demo', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def process_pic_without_shape_change(self, path, is_gauss_blur=True,
                                         is_gauss_noise=True):
        image = self.read_image(path)
        try:
            shape = image.shape
        except:
            print("{} file is broken".format(path))
            return

        if is_gauss_blur:
            image = self.gauss_blur(image)
        if is_gauss_noise:
            image = self.gauss_noise(image)

        return image

    def process_pic_with_shape_change(self, path, is_shear_mapping=False, is_wrap_affine=True,
                                      is_projective_transform=True,
                                      is_motion_blur=False, is_gauss_blur=True,
                                      is_gauss_noise=True):
        image = self.read_image(path)
        try:
            shape = image.shape
        except:
            print("{} file is broken".format(path))
            return
        if is_shear_mapping:
            image = self.shear_mapping(image)
        if is_wrap_affine:
            image = self.wrap_affine(image)
        if is_projective_transform:
            image = self.projective_transform(image)
        if is_motion_blur:
            image = self.motion_blur(image)
        if is_gauss_blur:
            image = self.gauss_blur(image)
        if is_gauss_noise:
            image = self.gauss_noise(image)

        return image

    # 对某个图像的区域进行扣取
    # 2018.03.06 这个之后写
    def crop_image(self, path, box=(0, 10, 10, 10)):
        image = self.read_image(path)
        pass


pp = plate_process()
# image = pp.read_image('./云0B42KH.png')
# image = pp.change_light(image, 1.2)
# pp.show_image(image)
# pp.process_pic_without_shape_change('./云0B42KH.png',
#                                     is_gauss_blur=False,
#                                     is_gauss_noise=False)

import concurrent.futures
folder = '../../plate_dataset/license'
save_folder = '../../plate_dataset/plate_process_image_without_shape'

def exec(name):
    range_list = []
    for j in range(2):
        if random.random() < 0.5:
            range_list.append(True)
        else:
            range_list.append(False)
    range_list.append(True)
    path = os.path.join(folder, name)
    # image = pp.process_pic_with_shape_change(path, range_list[0], range_list[1], range_list[2], range_list[3], range_list[4], range_list[5])
    image = pp.process_pic_without_shape_change(path, range_list[0], range_list[1])
    pp.save_image(image, name, save_folder)
    return path

names = os.listdir(folder)
with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
    for i, p in enumerate(executor.map(exec, names)):
        print(i, p)
