import tensorflow as tf
import numpy as np
from tensorflow.python import debug as tf_debug


log_dir = "./save/logs"

x = tf.placeholder(tf.float32, [None, 5], name='x')
w = tf.Variable(tf.truncated_normal([5, 2]), name='w')

b = tf.Variable(tf.zeros(2), name='b')

logits = x @ w + b
y = tf.nn.softmax(logits)

y_ = tf.placeholder(tf.float32, [None, 2], name="labels")

loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(y_, logits), name='loss')

optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

writer = tf.summary.FileWriter(log_dir + "/train", sess.graph)

saver = tf.train.Saver()
saver.save(sess, log_dir + "/model.ckpt")

sess = tf_debug.TensorBoardDebugWrapperSession(sess, "localhost:6065")


train_x = np.array([[1, 2, 3, 4, 5],
                    [2, 3, 4, 5, 6],
                    [3, 4, 5, 6, 7]])

train_y = np.array([[1, 0],
                    [1, 0],
                    [0, 1]])

for i in range(10):
    sess.run((optimizer, loss), feed_dict={x: train_x,
                                           y_: train_y})
