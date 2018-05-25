import os, cv2
import tensorflow as tf
import matplotlib.image as img
import matplotlib.pyplot as plt
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

sess = tf.InteractiveSession()
image = cv2.imread('10.png')
reshaped_image = tf.cast(image,tf.float32)
size = tf.cast(tf.shape(reshaped_image).eval(),tf.int32)
height = sess.run(size[0])
width = sess.run(size[1])
# distorted_image = tf.random_crop(reshaped_image,[height - np.random.randint(0, 50), width - np.random.randint(0, 50), 3])
distorted_image = tf.random_crop(reshaped_image,[height - 40, width - np.random.randint(0, 50), 3])
distorted_image = tf.image.resize_images(distorted_image, (height, width))
print(tf.shape(reshaped_image).eval())
print(tf.shape(distorted_image).eval())

fig = plt.figure()
fig1 = plt.figure()
ax = fig.add_subplot(111)
ax1 = fig1.add_subplot(111)

ax.imshow(sess.run(tf.cast(reshaped_image,tf.uint8)))
ax1.imshow(sess.run(tf.cast(distorted_image,tf.uint8)))
plt.show()