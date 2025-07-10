import tensorflow as tf
from tensorflow.keras.layers import Conv2D

class SelfAttention(tf.keras.layers.Layer):
  def __init__(self, filters, **kwargs):
    super(SelfAttention, self).__init__(**kwargs)
    self.filters = filters
    self.query = Conv2D(filters // 8, kernel_size=1, padding="same")
    self.key = Conv2D(filters // 8, kernel_size=1, padding="same")
    self.value = Conv2D(filters, kernel_size=1, padding="same")
    self.gamma = tf.Variable(initial_value=tf.zeros(1), trainable=True)

  def call(self, x):
    batch, height, width, channels = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[-1]
    q = tf.reshape(self.query(x), (batch, height * width, channels // 8))
    k = tf.reshape(self.key(x), (batch, height * width, channels // 8))
    v = tf.reshape(self.value(x), (batch, height * width, channels))
    attn = tf.nn.softmax(tf.matmul(q, k, transpose_b=True))
    attn_out = tf.matmul(attn, v)
    attn_out = tf.reshape(attn_out, (batch, height, width, channels))
    return self.gamma * attn_out + x