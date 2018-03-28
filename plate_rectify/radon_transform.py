import matplotlib.pyplot as plt

from skimage import io, filters, color
from skimage import data_dir
from skimage import transform
from scipy.ndimage import zoom
import time

t1 = time.time()
image = io.imread("./s4.png")
# ico = zoom(image, 0.4)
gray = color.rgb2gray(image)
sobel = filters.sobel(gray)

# image = zoom(image, 0.4)
t2 = time.time()
r = transform.radon(sobel)
print("radon transform cost time: {}s".format(time.time() - t2))
# print(r.shape)
(m, n) = r.shape
c = 1
for i in range(m):
    for j in range(n):
        if r[0, 0] < r[i, j]:
            r[0, 0] = r[i, j]
            c = j
angle = 90 - c
dst = transform.rotate(image, angle)
print("cost time: {}s".format(time.time() - t1))

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
plt.title("radon")
plt.imshow(r)

plt.subplot(325)
plt.title("dst")
plt.imshow(dst)

plt.subplots_adjust(hspace=0.6, wspace=0.5)
plt.show()
plt.close()
