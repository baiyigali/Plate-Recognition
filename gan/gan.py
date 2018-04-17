import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/Users/Nelson/Desktop/Computer/zhihu/denoise_auto_encoder/MNIST_data/')

img = mnist.train.images[50]
plt.imshow(img.reshape((28, 28)), cmap='Greys_r')

