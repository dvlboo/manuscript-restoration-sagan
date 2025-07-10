import tensorflow as tf

# 🌟 Spectral Normalization Layer
class SpectralNormalization(tf.keras.layers.Wrapper):
	def __init__(self, layer, power_iterations=1, **kwargs):
		super(SpectralNormalization, self).__init__(layer, **kwargs)
		self.power_iterations = power_iterations
		self.u = None  # defer creation to build()

	def build(self, input_shape):
		if not self.layer.built:
			self.layer.build(input_shape)
		self.kernel = self.layer.kernel
		if self.u is None:
			self.u = self.add_weight(
				shape=(1, self.kernel.shape[-1]),
				initializer="random_normal",
				trainable=False,
				name="sn_u"
			)
		super().build(input_shape)

	def call(self, inputs, training=None):
		# Power iteration
		w_shape = self.kernel.shape.as_list()
		w = tf.reshape(self.kernel, [-1, w_shape[-1]])
		u = self.u
		for _ in range(self.power_iterations):
			v = tf.linalg.l2_normalize(tf.matmul(u, tf.transpose(w)))
			u = tf.linalg.l2_normalize(tf.matmul(v, w))
		sigma = tf.matmul(tf.matmul(v, w), tf.transpose(u))
		w_norm = w / sigma
		self.layer.kernel.assign(tf.reshape(w_norm, w_shape))
		self.u.assign(u)
		return self.layer(inputs, training=training)