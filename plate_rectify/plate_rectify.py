# encoding=utf-8
import cv2
import numpy as np
import math
import time
import matplotlib.pyplot as plt


class plate_rectify():
    def __init__(self):
        pass

    def show_image(self, image, name_window="1"):
        cv2.namedWindow(name_window)
        cv2.imshow(name_window, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def process(self, image):
        # self.show_image(image)
        r, c, _ = image.shape
        self.image_area = r * c
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        # h, s, v = cv2.split(hsv)
        # self.show_image(v)
        # self.show_image(gray)
        gaussian = cv2.GaussianBlur(gray, (3, 3), 0, 0, cv2.BORDER_DEFAULT)
        # 中值滤波
        median = cv2.medianBlur(gaussian, 5)
        self.show_image(median)
        # Sobel算子，X方向求梯度
        sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
        self.show_image(sobel)
        # 二值化
        ret, binary = cv2.threshold(sobel, 170, 255, cv2.THRESH_BINARY)
        self.show_image(binary)
        element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
        element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
        # 膨胀一次，让轮廓突出
        dilation = cv2.dilate(binary, element2, iterations=1)
        self.show_image(dilation)
        # 腐蚀一次，去掉细节
        erosion = cv2.erode(dilation, element1, iterations=1)
        self.show_image(erosion)
        # 再次膨胀，让轮廓明显一些
        dilation2 = cv2.dilate(erosion, element2, iterations=3)
        return dilation2

    def hough_method(self, path, threshold=100):
        image = cv2.imread(path)
        dst = image.copy()
        r, c, _ = image.shape
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=1)

        lines = cv2.HoughLines(sobel, 1, np.pi / 180, threshold=threshold)
        print(lines.__len__())
        for line in lines:
            rho = line[0][0]
            theta = line[0][1]
            print(rho, theta, math.degrees(theta))
            angle = theta * 180 / np.pi
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(dst, (x1, y1), (x2, y2), (255, 0, 0), 1)
        rotated = self.rotate(dst, angle)

        plt.figure()
        plt.subplot(321)
        plt.title("source image")
        plt.imshow(image)

        plt.subplot(322)
        plt.title("gray image")
        plt.imshow(gray)

        plt.subplot(323)
        plt.title("sobel")
        plt.imshow(sobel)

        plt.subplot(324)
        plt.title("dst")
        plt.imshow(dst)

        plt.subplot(325)
        plt.title("rotated")
        plt.imshow(rotated)

        plt.subplots_adjust(hspace=0.6, wspace=0.5)
        plt.show()
        plt.close()

        # angle = self.get_angle(x1, y1, x2, y2)
        # print(angle)
        # dst = self.rotate(image, angle)

        # self.show_image(dst, "image")

    def draw_box(self, image, x1, y1, x2, y2):
        dst = cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        dst = cv2.line(dst, (x1, y1), (x2, y2), (0, 255, 0))
        self.show_image(dst)

    def rectify(self, path):
        image = cv2.imread(path)

        dilation2 = self.process(image)
        self.show_image(dilation2)
        _, contours, hierarchy = cv2.findContours(dilation2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        region = []
        for i in range(len(contours)):
            cnt = contours[i]
            area = cv2.contourArea(cnt)
            if area < self.image_area / 10.:
                continue
            rect = cv2.minAreaRect(cnt)
            theta = rect[2]
            print(theta)
            # self.show_image(image)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            print(box)
            self.draw_box(image, box[1][0], box[1][1], box[3][0], box[3][1])
            if theta < -20:
                theta += 90

            dst = self.rotate(image, theta)
            self.show_image(dst)

    def rotate(self, image, angle, center=None, scale=1.0):
        if angle > 90:
            angle = 180 - angle
            # 获取图像尺寸
        (h, w) = image.shape[:2]

        # 若未指定旋转中心，则将图像中心设为旋转中心
        if center is None:
            center = (w / 2, h / 2)

        # 执行旋转
        M = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.warpAffine(image, M, (w, h))

        # 返回旋转后的图像
        return rotated

    def get_angle(self, x1, y1, x2, y2):
        delta_x = x1 - x2
        delta_y = y1 - y2
        theta = math.atan(delta_y / delta_x) * 180 / np.pi
        return theta

    def radon(self):
        pass


if __name__ == '__main__':
    t1 = time.time()
    pr = plate_rectify()
    pr.hough_method('s3.png', 100)
    print("cost time {:.6f}ms".format((time.time() - t1) * 1000))
