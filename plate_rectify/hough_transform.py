import matplotlib.pyplot as plt

from skimage import io, filters, color
from skimage import data_dir
from skimage import transform
from scipy.ndimage import zoom
import time
import numpy as np

t1 = time.time()
image = io.imread("./s4.png")
# ico = zoom(image, 0.4)
gray = color.rgb2gray(image)
guassion = filters.gaussian(gray)
median = filters.median(guassion)
sobel = filters.sobel_h(gray)

h, theta, d = transform.hough_line(sobel)

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

row, col = gray.shape

# for i, hlp in enumerate(transform.hough_line_peaks(h, theta, d)):
#     print(i, type(hlp))
hlp = transform.hough_line_peaks(h, theta, d)

for angle, dist in zip(hlp[1], hlp[2]):
    ag = angle * 180 / np.pi
    # print(ag)
    # if abs(angle) > 1.5:
    if abs(ag) > 80:
        print(ag)
        y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
        y1 = (dist - col * np.cos(angle)) / np.sin(angle)
        plt.plot((0, col), (y0, y1))
plt.imshow(image)

plt.subplots_adjust(hspace=0.6, wspace=0.5)
plt.show()
plt.close()
