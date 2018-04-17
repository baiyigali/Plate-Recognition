# encoding=utf-8
import sys

sys.path.append('..')
sys.path.append('../..')
import numpy as np
import cv2
import math
import random
import os
import concurrent.futures

PROJECT_PATH = '/home1/fsb/project/LPR/Plate-Recognition'


# 对图像进行错切变换、仿射变换、投影变换、运动模糊、高斯模糊、噪声处理
#
class plate_process():
    def __init__(self):
        pass

    # 错切变换
    def _shear_mapping(self, image, angel=None, max_angel=10):
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

    def _shear_mapping2(self, image, offset=None):
        rows, cols, _ = image.shape
        pts1 = np.float32([[0, 0], [0, rows], [cols, 0], [cols, rows]])
        rand_xy = random.random()
        rand_s = random.random()
        if offset is None:
            offset = random.randrange(int(-rows / 30.), int(rows / 10.))
        size = (cols, rows)
        if rand_xy < 0.5:  # 沿x轴错切
            pts2 = np.float32(
                [[0 + offset, 0], [0 - offset, rows], [cols + offset, 0], [cols - offset, rows]])  # 固定B, C点不动
        else:  # 沿y轴错切
            pts2 = np.float32(
                [[0, 0 + offset], [0, rows + offset], [cols, 0 - offset], [cols, rows - offset]])  # 固定C, D点不动

        M = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(image, M, size)

        return dst

    def _random_int(self, a=-10, b=10):
        return random.randint(a, b)

    # 仿射变换
    def _wrap_affine(self, image):
        rows, cols, _ = image.shape

        def up():
            # rd = random.randint(5, 20)
            srcpoint = np.float32([[0, 0], [0, rows], [cols, rows]])
            canvas = np.float32([[0 + self._random_int(), 0 + self._random_int()],
                                 [0 + self._random_int(), rows + self._random_int()],
                                 [cols + self._random_int(), rows + self._random_int()]])
            return srcpoint, canvas

        def right():
            rd = random.randint(5, 20)
            srcpoint = np.float32([[0, 0], [cols, 0], [cols, rows]])
            canvas = np.float32([[0 + self._random_int(), 0 + self._random_int()],
                                 [cols + self._random_int(), 0 + self._random_int()],
                                 [cols + self._random_int(), rows + self._random_int()]])
            return srcpoint, canvas

        def down():
            rd = random.randint(5, 20)
            srcpoint = np.float32([[cols, 0], [0, rows], [cols, rows]])
            canvas = np.float32([[cols + self._random_int(), 0 + self._random_int()],
                                 [0 + self._random_int(), rows + self._random_int()],
                                 [cols + self._random_int(), rows + self._random_int()]])
            return srcpoint, canvas

        def left():
            rd = random.randint(5, 20)
            srcpoint = np.float32([[0, 0], [0, rows], [cols, 0]])
            canvas = np.float32([[0 + self._random_int(), 0 + self._random_int()],
                                 [0 + self._random_int(), rows + self._random_int()],
                                 [cols + self._random_int(), 0 + self._random_int()]])
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
    def _projective_transform(self, image):
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
    def _motion_blur(self, image, degree=None, angle=None):
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
    def _gauss_blur(self, image, degree=12):
        if degree is None:
            degree = random.randint(8, 12)

        return cv2.blur(image, (degree, degree))

    # 噪声
    def _gauss_noise(self, image, degree=None):
        row, col, ch = image.shape
        mean = 0
        if degree is None:
            degree = random.uniform(20, 60)

        sigma = degree ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss
        cv2.normalize(noisy, noisy, 0, 255, norm_type=cv2.NORM_MINMAX)
        dst = np.array(noisy, dtype=np.uint8)
        return dst

    # 图片高亮
    def _change_light(self, image, degree=1.02):
        row, col, ch = image.shape
        dst = image * degree
        return dst

    # 生成遮挡物
    def _add_blocked(self, image):
        height, width, _ = image.shape
        nums = np.random.random_integers(0, 10)  # 随机生成遮挡物的数量
        for i in range(nums):
            size = np.random.random_integers(20, 50)  # 随机生成遮挡物尺寸
            x = np.random.random_integers(0, width - size)  # 随机生成坐标
            y = np.random.random_integers(0, height - size)
            block = np.array(np.random.randint(0, 255, size ** 2 * 3)).reshape((size, size, 3))
            image[y:y + size, x:x + size, :] = block
        return image

    # 生成颜色遮挡 模拟高光???
    def _add_color(self, image):
        height, width, _ = image.shape
        nums = np.random.random_integers(0, 3)  # 随机生成遮挡物的数量
        for i in range(nums):
            size = np.random.random_integers(50, 100)  # 随机生成遮挡物尺寸
            x = np.random.random_integers(0, width - size)  # 随机生成坐标
            y = np.random.random_integers(0, height - size)
            block = image[y:y + size, x:x + 2 * size, :]
            if np.random.random() < 0.5:
                color = np.full_like(block, np.random.randint(0, 254))
            else:
                color = np.full_like(block, 255)
            alpha = np.random.randint(1, 7) / 10.
            gamma = np.random.uniform(0.1, 4)
            image[y:y + size, x:x + 2 * size, :] = cv2.addWeighted(block, alpha, color, 1 - alpha, gamma)
        return image

    def _hist_euqalize(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        dst = cv2.equalizeHist(image)
        return dst

    def _gamma_correction(self, image, gamma=None):
        if gamma is None:
            gamma = np.random.uniform(0.9, 1.1)
        dst = np.power(image / 255.0, gamma) * 255.0
        return dst

    def _move(self, image):
        x = np.random.randint(-10, 10)
        y = np.random.randint(-10, 10)
        M = np.float32([[1, 0, x], [0, 1, y]])
        dst = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        return dst

    def _zoom(self, image, coords):

        row, col, _ = image.shape
        bg_pic = os.listdir(os.path.join(PROJECT_PATH, 'image/car'))
        back_ground = cv2.imread(os.path.join(PROJECT_PATH, "./image/car/") + np.random.choice(bg_pic, 1)[0])
        dst = cv2.resize(back_ground, (col, row))
        # dst = np.zeros_like(image)
        scale = np.random.randint(9, 11) / 10.
        methods = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA,
                   cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]
        pic = image
        resized = cv2.resize(src=pic, dsize=(int(col * scale), int(row * scale)),
                             interpolation=np.random.choice(methods, 1)[0])
        # self._show_image(resized)
        new_row, new_col, _ = resized.shape
        if row > new_row:
            x = int((row - new_row) / 2)
            y = int((col - new_col) / 2)
            dst[x:x + new_row, y:y + new_col, :] = resized
        else:
            x = int((new_row - row) / 2)
            y = int((new_col - col) / 2)
            dst = resized[x:x + row, y:y + col, :]

        # 根据比例更改坐标系的值
        if coords is not None:
            for i, coord in enumerate(coords):
                coord[0:2] = coord[0:2] * scale + (1 - scale) / 2
                coord[2:4] = coord[2:4] * scale

        return dst, coords

    def _scale(self, image, dst_row=60):
        row, col, _ = image.shape
        first_row = 20.
        dst = cv2.resize(image, dsize=(int(first_row * col / row), int(first_row)))
        dst = cv2.resize(dst, dsize=(int(dst_row * col / row), int(dst_row)))
        return dst

    def _read_image(self, path):
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)

    def _save_image(self, image, name, folder='./'):
        if not os.path.exists(folder):
            os.makedirs(folder)
        path = os.path.join(folder, name)
        cv2.imwrite(path, image)
        print("save image at {}".format(path))

    def _show_image(self, image):
        cv2.namedWindow('demo')
        cv2.imshow('demo', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def process_pic_with_shape_change(self, image,
                                      coords=None,
                                      is_zoom=True,
                                      is_shear_mapping=True,
                                      is_gauss_blur=True,
                                      is_gauss_noise=True,
                                      is_add_white=True,
                                      is_gamma_correction=False,
                                      is_move=True,
                                      is_scale=True):
        if image.shape is not None:
            if is_zoom: image, coords = self._zoom(image, coords)

            if is_shear_mapping: image = self._shear_mapping2(image)

            if is_gauss_blur: image = self._gauss_blur(image)

            if is_gauss_noise: image = self._gauss_noise(image)

            # if is_add_block: image = self.add_blocked(image)
            if is_add_white: image = self._add_color(image)

            if is_gamma_correction: image = self._gamma_correction(image)

            if is_move: image = self._move(image)
            if is_scale: image = self._scale(image)
            image = cv2.resize(image, (100, 30)) #  to (100, 30)

        return image, coords

    # 对某个图像的区域进行扣取
    # 2018.03.06 这个之后写
    def crop_image(self, path, box=(0, 10, 10, 10)):
        image = self._read_image(path)
        pass


if __name__ == "__main__":
    pp = plate_process()
    # image = pp._read_image('./云0B42KH.png')
    # # image = pp._change_light(image, 1.2)
    # image = pp._hist_euqalize(image)
    # # image = pp._gamma_corection(image)
    # image = pp._move(image)
    # image = pp._zoom(image)
    # image = pp._scale(image)
    # pp._show_image(image)
    path = './云0B42KH.png',
    image = cv2.imread(path)
    image = pp.process_pic_with_shape_change(image)
    pp._show_image(image)
    # pp._save_image(image, '1.png', './')

    # # batch process
    # folder = '../../plate_dataset_new/license'
    # save_folder = '../../plate_dataset_new/plate_image_resized'
    #
    # names = os.listdir(folder)
    #
    #
    # def exec(name):
    #     range_list = []
    #     for j in range(7):
    #         if random.random() < 0.5:
    #             range_list.append(True)
    #         else:
    #             range_list.append(False)
    #
    #     path = os.path.join(folder, name)
    #     image = cv2.imread(path)
    #     image = pp.process_pic_with_shape_change(image, range_list[0], range_list[1], range_list[2], range_list[3],
    #                                              range_list[4], range_list[5])
    #     pp._save_image(image, name, save_folder)
    #     return path, range_list
    #
    #
    # with concurrent.futures.ThreadPoolExecutor(max_workers=80) as executor:
    #     for i, (p, l) in enumerate(executor.map(exec, names)):
    #         print(i, p, l)
