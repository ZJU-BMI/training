import tensorflow as tf
import numpy as np


def xavier_init(fan_in, fan_out, constant=1):
    """这是对于权重矩阵的初始化
    一个权重W的维度为 [fan_in, fan_out]，则其中一共有 fan_in * fan_out个元素
    这些元素的分布满足在 [-sqrt (6 / (fan_in + fan_out))  sqrt(6 / (fan_in + fan_out))]上的均匀分布

    """
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high)


class LR(object):
    def __init__(self, n_input, n_class, l2=1e-8, lr=0.001, batch_size=64, epochs=10000):
        self._n_input = n_input
        self._n_class = n_class
        self._l2 = l2
        self._lr = lr
        self._batch_size = batch_size
        self._epochs = epochs

        self._x = tf.placeholder(tf.float32, [None, self._n_input], 'x_input')

        if self._n_class == 2:
            self._y_ = tf.placeholder(tf.float32, [None, 1], 'y_true')
            self.weights = {
                "weight": tf.Variable(xavier_init(self._n_input, 1), dtype=tf.float32, name='w'),
                "bias": tf.Variable(tf.zeros(1), name='b1')
            }
            self._y = self._x @ self.weights['weight'] + self.weights['bias']
            self._pred = tf.nn.sigmoid(self._y)

            self._l2_loss = tf.nn.l2_loss(self.weights['weight'], name='l2_loss')
            self._cross_loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(self._y_, self._y))
        else:
            self._y_ = tf.placeholder(tf.float32, [None, self._n_class], 'y_true')
            self.weights = {
                "weight": tf.Variable(xavier_init(self._n_input, self._n_class), dtype=tf.float32, name='w'),
                "bias": tf.Variable(tf.zeros(self._n_class), name='b1')
            }
            self._y = self._x @ self.weights['weight'] + self.weights['bias']
            self._pred = tf.nn.softmax(self._y)

            self._l2_loss = tf.nn.l2_loss(self.weights['weight'], name='l2_loss')
            self._cross_loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(self._y_, self._y), name='loss')

        self._loss = self._cross_loss + self._l2 * self._l2_loss
        self.optimizer = tf.train.AdamOptimizer(self._lr).minimize(self._loss)

        self._sess = tf.Session()

        self._writer = tf.summary.FileWriter("./save/lr", self._sess.graph)
        self._sess.run(tf.global_variables_initializer())
        self._saver = tf.train.Saver()
        self._saver.save(self._sess, './save/lr/lr.ckpt')

    def fit(self, data_set):
        init = tf.global_variables_initializer()
        self._sess.run(init)

        for i in range(self._epochs):
            x, y = data_set.next_batch(self._batch_size)
            self._sess.run(self.optimizer, feed_dict={self._x: x, self._y_: y})

            if i % 500 == 0:
                loss = self._sess.run(self._loss, feed_dict={self._x: data_set.examples, self._y_: data_set.labels})
                print(i, loss)

    def predict(self, x):
        return self._sess.run(self._pred, feed_dict={self._x: x})


class Conv(object):
    def __init__(self):
        with tf.variable_scope('conv'):
            x = tf.placeholder(tf.float32, [None, 28, 28, 1], 'input')

            y = tf.contrib.layers.conv2d(x, 3, [28, 3], padding='VALID')

            self._sess = tf.Session()
            self._sess.run(tf.global_variables_initializer())
            self._writer = tf.summary.FileWriter('./save/conv', self._sess.graph)
            self._saver = tf.train.Saver()
            self._saver.save(self._sess, './save/conv/conv.ckpt')


if __name__ == "__main__":
    conv = Conv()
