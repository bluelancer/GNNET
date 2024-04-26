from tf_geometric.nn.conv.gcn import gcn, gcn_build_cache_for_graph
import tensorflow as tf
import warnings



class DGCN(tf.keras.Model):
	"""
	Graph Convolutional Layer
	"""

	def build(self, input_shapes):
		x_shape = input_shapes[0]
		num_features = x_shape[-1]
		# import ipdb; ipdb.set_trace()	
		self.kernel1 = self.add_weight("kernel1", shape=[num_features, self.units],
									  initializer="glorot_uniform", regularizer=self.kernel_regularizer)
		self.kernel2in = self.add_weight("kernel2in", shape=[num_features, self.units],
									  initializer="glorot_uniform", regularizer=self.kernel_regularizer)
		self.kernel2out = self.add_weight("kernel2out", shape=[num_features, self.units],
									  initializer="glorot_uniform", regularizer=self.kernel_regularizer)
		if self.use_bias:
			self.bias1 = self.add_weight("bias", shape=[self.units],
										initializer="zeros", regularizer=self.bias_regularizer)
			self.bias2in = self.add_weight("bias2in", shape=[self.units],
						initializer="zeros", regularizer=self.bias_regularizer)
			self.bias2out = self.add_weight("bias2out", shape=[self.units],
						initializer="zeros", regularizer=self.bias_regularizer)
		self.concate_weight2in = self.add_weight("concate_weight2in", shape=[1],
									  initializer="glorot_uniform", regularizer=self.kernel_regularizer)
		self.concate_weight2out = self.add_weight("concate_weight2out", shape=[1],
									  initializer="glorot_uniform", regularizer=self.kernel_regularizer)
		self.fc_layer = tf.keras.Sequential([
			tf.keras.layers.InputLayer(input_shape=3 * num_features),
			tf.keras.layers.Dense(num_features,activation=tf.nn.relu, use_bias=True)
		])
	def __init__(self, units, activation=tf.nn.relu,
				 use_bias=True,
				 renorm=True, improved=False,
				 kernel_regularizer=None, bias_regularizer=None, *args, **kwargs):
		"""
		:param units: Positive integer, dimensionality of the output space.
		:param activation: Activation function to use.
		:param use_bias: Boolean, whether the layer uses a bias vector.
		:param renorm: Whether use renormalization trick (https://arxiv.org/pdf/1609.02907.pdf).
		:param improved: Whether use improved GCN or not.
		:param kernel_regularizer: Regularizer function applied to the `kernel` weights matrix.
		:param bias_regularizer: Regularizer function applied to the bias vector.
		"""
		super().__init__(*args, **kwargs)
		self.units = units

		self.activation = activation
		self.use_bias = use_bias

		self.renorm = renorm
		self.improved = improved

		self.kernel_regularizer = kernel_regularizer
		self.bias_regularizer = bias_regularizer

	def build_cache_for_graph(self, graph, override=False):
		"""
		Manually compute the normed edge based on this layer's GCN normalization configuration (self.renorm and self.improved) and put it in graph.cache.
		If the normed edge already exists in graph.cache and the override parameter is False, this method will do nothing.
		:param graph: tfg.Graph, the input graph.
		:param override: Whether to override existing cached normed edge.
		:return: None
		"""
		gcn_build_cache_for_graph(graph, self.renorm, self.improved, override=override)

	def cache_normed_edge(self, graph, override=False):
		"""
		Manually compute the normed edge based on this layer's GCN normalization configuration (self.renorm and self.improved) and put it in graph.cache.
		If the normed edge already exists in graph.cache and the override parameter is False, this method will do nothing.
		:param graph: tfg.Graph, the input graph.
		:param override: Whether to override existing cached normed edge.
		:return: None
		.. deprecated:: 0.0.56
			Use ``build_cache_for_graph`` instead.
		"""
		warnings.warn("'GCN.cache_normed_edge(graph, override)' is deprecated, use 'GCN.build_cache_for_graph(graph, override)' instead", DeprecationWarning)
		return self.build_cache_for_graph(graph, override=override)

	def call(self, inputs, cache=None, training=None, mask=None):
		"""
		:param inputs: List of graph info: [x, edge_index, edge_weight]
		:param cache: A dict for caching A' for GCN. Different graph should not share the same cache dict.
		:return: Updated node features (x), shape: [num_nodes, units]
		"""

		assert len(inputs) == 7

		x, edge_index_1st_prox, edge_weight_1st_prox, edge_index_2nd_prox_in, edge_weight_2nd_prox_in, edge_index_2nd_prox_out, edge_weight_2nd_prox_out  = inputs

		y_1st_prox=gcn(x, edge_index_1st_prox, edge_weight_1st_prox, self.kernel1, self.bias1,
					   activation=self.activation, renorm=self.renorm, improved=self.improved, cache=cache)
		y_2nd_in_prox=gcn(x, edge_index_2nd_prox_in, edge_weight_2nd_prox_in, self.kernel2in, self.bias2in,
					   activation=self.activation, renorm=self.renorm, improved=self.improved, cache=cache)
		y_2nd_out_prox=gcn(x, edge_index_2nd_prox_out, edge_weight_2nd_prox_out, self.kernel2out, self.bias2out,
					   activation=self.activation, renorm=self.renorm, improved=self.improved, cache=cache)
		concate_output = tf.keras.layers.concatenate([y_1st_prox,self.concate_weight2in*y_2nd_in_prox,self.concate_weight2out*y_2nd_out_prox], axis = 1)
		output = self.fc_layer(concate_output)		
		return output
