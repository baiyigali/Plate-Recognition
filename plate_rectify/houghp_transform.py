import matplotlib.pyplot as plt

from skimage import io, filters, color
from skimage import data_dir
from skimage import transform, feature
from scipy.ndimage import zoom
import time
import numpy as np

t1 = time.time()
image = io.imread("./s4.png")
# ico = zoom(image, 0.4)
gray = color.rgb2gray(image)
guassion = filters.gaussian(gray)
median = filters.median(guassion)
sobel = filters.sobel(median)
canny = feature.canny(gray)

lines = transform.probabilistic_hough_line(sobel, threshold=10, line_length=50, line_gap=3)

print("cost time: {}s".format(time.time() - t1))

plt.figure()
plt.subplot(321)
plt.title("source image")
plt.imshow(image)

plt.subplot(322)
plt.title("gray image")
plt.imshow(gray)

plt.subplot(323)
plt.title("guassion image")
plt.imshow(gray)

plt.subplot(324)
plt.title("median image")
plt.imshow(median)

plt.subplot(325)
plt.title("sobel")
plt.imshow(sobel)

plt.subplot(326)
plt.title("radon")


for line in lines:
    p0, p1 = line
    plt.plot((p0[0], p1[0]), (p0[0], p1[1]))
plt.imshow(image)

plt.subplots_adjust(hspace=0.6, wspace=0.5)
plt.show()
plt.close()
