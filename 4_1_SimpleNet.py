# coding: utf-8
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets

(x, y) ,(x_val, y_val) = datasets.mnist.load_data()
x = 2*tf.convert_to_tensor(x, dtype=tf.float32)/255
y = tf.convert_to_tensor(y, dtype = tf.int32)
y_onehot = tf.one_hot(y, depth=10)

# initial para of net
w1 = tf.variable(tf.random.truncated_normal([784, 256], stddev=0.1))
b1 = tf.variable(tf.zeros([256]))
w2 = tf.variable(tf.random.truncated_normal([256, 128], stddev=0.1))
b2 = tf.variable(tf.zeros([128]))
w3 = tf.variable(tf.random.truncated_normal([128, 10], stddev=0.1))
b3 = tf.variable(tf.zeros[10])


with tf.GradientTape() as tape:
    x = tf.reshape(x, [-1, 28 * 28])

    h1 = tf.matmul(x, w1)+b1
    h1 = tf.nn.relu(h1)
    h2 = tf.matmul(h1, w2)+b2
    h2 = tf.nn.relu(h2)
    out = tf.matmul(h2, w3)+b3

    loss = tf.square(y_onehot - out)
    loss = tf.reduce_mean(loss)/x.shape[0]

grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])

w1.assign_sub(lr * grads[0])
b1.assign_sub(lr * grads[1])
w2.assign_sub(lr * grads[2])
b2.assign_sub(lr * grads[3])
w3.assign_sub(lr * grads[4])
b3.assign_sub(lr * grads[5])

