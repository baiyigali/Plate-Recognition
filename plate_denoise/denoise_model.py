import tensorflow as tf


class denoise_autoencoder():
    def __init__(self, noise_images):
        self.noise_images = noise_images
        pass

    def network_model(self, scope='dae', alpha=0.1):
        with tf.variable_scope(scope):
            x = self.noise_images

            # encoder
            x = tf.layers.conv2d(x, filters=64, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu,
                                 name='conv_1')
            x = tf.layers.max_pooling2d(x, pool_size=3, strides=2, padding='same', name='pool_1')
            x = tf.layers.conv2d(x, filters=128, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu,
                                 name='conv_2')
            x = tf.layers.conv2d(x, filters=256, kernel_size=1, strides=1, padding='same', activation=tf.nn.relu,
                                 name='conv_3')
            # decoder
            x = tf.layers.conv2d(x, filters=128, kernel_size=1, strides=1, padding='same', activation=tf.nn.relu,
                                 name='conv_4')
            x = tf.image.resize_nearest_neighbor(x, (15, 50))
            x = tf.layers.conv2d(x, filters=64, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu,
                                 name='conv_5')
            x = tf.image.resize_nearest_neighbor(x, (30, 100))
            x = tf.layers.conv2d(x, filters=3, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu,
                                 name='conv_6')

        return x

    def _leaky_relu(self, alpha):
        def op(inputs):
            return tf.maximum(alpha * inputs, inputs, name='leaky_relu')

    def loss(self, predicts, labels):
        with tf.variable_scope('mse_loss'):
            loss = tf.clip_by_value(predicts, 1e-10, 1) - tf.clip_by_value(labels, 1e-10, 1)
            loss = tf.reduce_mean(tf.reduce_sum(tf.square(loss), axis=[1, 2, 3]), name='loss')
            tf.summary.scalar('loss', loss)
        # with tf.variable_scope('ce_loss'):
        #     predicts = tf.clip_by_value(predicts, 1e-10, 1.0)
        #     labels = tf.clip_by_value(labels, 1e-10, 1.0)
        #     print(tf.reduce_sum(predicts * tf.log(labels), axis=[1, 2, 3]))
        #     loss = -tf.reduce_mean(tf.reduce_sum(predicts * tf.log(labels), axis=[1, 2, 3]))
        return loss

    def accuracy(self):
        pass


if __name__ == '__main__':
    noise_images = tf.placeholder(tf.float32, [32, 30, 100, 3])
    source_images = tf.placeholder(tf.float32, [32, 30, 100, 3])
    dae = denoise_autoencoder(noise_images)
    print(dae.loss(noise_images, source_images))
