"""
Copyright 2021 Universitat Politècnica de Catalunya & AGH University of Science and Technology
									BSD 3-Clause License
Redistribution and use in source and binary forms, with or without modification, are permitted
provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this list of conditions
and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of
conditions and the following disclaimer in the documentation and/or other materials provided
with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse
or promote products derived from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""
import tensorflow as tf
#from GNNLayer.GCNLayer import GraphConvLayer
import tf_geometric as tfg
from RouteRolxNet.nalu import NALU
from tf_geometric.layers import DropEdge
from tensorflow.keras import regularizers
from RouteRolxNet.dgcn import DGCN 

#from tfg_GraphSage_wrapper import Tfg_graphsage_mean

class RouteNetModel(tf.keras.Model):
	def __init__(self, config, output_units=1, Homo=None, combination_type=None):
		super(RouteNetModel, self).__init__()

		# Configuration dictionary. It contains the needed Hyperparameters for the model.
		# All the Hyperparameters can be found in the config.ini file
		self.config = config

		if self.config['HYPERPARAMETERS']['regularization'] == 'l2':
			print ('Note: Using l2 nomalization')
			gcn_kernel_regularizer = regularizers.l2(float(self.config['HYPERPARAMETERS']['regularization_index']))
			gcn_bias_regularizer = regularizers.l2(float(self.config['HYPERPARAMETERS']['regularization_index']))
			gat_kernel_regularizer = regularizers.l2(float(self.config['HYPERPARAMETERS']['regularization_index']))
			gat_bias_regularizer = regularizers.l2(float(self.config['HYPERPARAMETERS']['regularization_index']))
		else:
			gcn_kernel_regularizer = None
			gcn_bias_regularizer = None
			gat_kernel_regularizer = None
			gat_bias_regularizer = None
			
		self.link_embedding = tf.keras.Sequential([
			tf.keras.layers.InputLayer(input_shape=1),
			NALU(int(int(self.config['HYPERPARAMETERS']['link_state_dim']) / 2)),
			NALU(int(self.config['HYPERPARAMETERS']['link_state_dim']))
		])

		self.readout_link = tf.keras.Sequential([
			tf.keras.layers.InputLayer(input_shape=int(
				self.config['HYPERPARAMETERS']['link_state_dim'])),
			NALU(int(int(self.config['HYPERPARAMETERS']['readout_units']))),
			NALU(int(self.config['HYPERPARAMETERS']['readout_units'])),
			NALU(output_units)
		])
		
		if self.config['HYPERPARAMETERS']['dropedge'] == 'Yes':
			print ('Note: using Drop Edge')
			self.dropedge =  DropEdge(float(self.config['HYPERPARAMETERS']['edge_drop_rate']), force_undirected=True)

		self.dgcn0 = DGCN(
			int(self.config['HYPERPARAMETERS']['link_state_dim']),improved=True ,activation=tf.nn.relu, kernel_regularizer = gcn_kernel_regularizer, bias_regularizer = gcn_bias_regularizer)
		self.dgcn1 = DGCN(
			int(self.config['HYPERPARAMETERS']['link_state_dim']),improved=True ,activation=tf.nn.relu, kernel_regularizer = gcn_kernel_regularizer, bias_regularizer = gcn_bias_regularizer)
		self.dgcn2 = DGCN(
			int(self.config['HYPERPARAMETERS']['link_state_dim']),improved=True ,activation=tf.nn.relu,kernel_regularizer = gcn_kernel_regularizer, bias_regularizer = gcn_bias_regularizer)

		self.gat0 = tfg.layers.GAT(int(
			self.config['HYPERPARAMETERS']['link_state_dim']), activation=tf.nn.relu, num_heads=int(self.config['HYPERPARAMETERS']['attention_heads']), attention_units=int(self.config['HYPERPARAMETERS']['attention_units']), drop_rate=0.3,kernel_regularizer = gat_kernel_regularizer, bias_regularizer = gat_bias_regularizer)

		self.dropout = tf.keras.layers.Dropout(0.1)

		self.condense_net = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=int(4 * int(self.config['HYPERPARAMETERS']['link_state_dim']))), tf.keras.layers.Dense(2 * int(self.config['HYPERPARAMETERS']['link_state_dim']), use_bias=True),
												tf.keras.layers.Dense(int(self.config['HYPERPARAMETERS']['link_state_dim']), use_bias=True)])

	@tf.function
	def call(self, inputs):
		"""This function is execution each time the model is called
		Args:
			inputs (dict): Features used to make the predictions.
		Returns:
			tensor: A tensor containing the per-path delay.
		"""
		# import ipdb; ipdb.set_trace()
		traffic = tf.expand_dims(tf.squeeze(inputs['traffic']), axis=1)
		capacity = tf.expand_dims(tf.squeeze(inputs['capacity']), axis=1)

		link_to_path = tf.squeeze(inputs['link_to_path'])

		path_to_link = tf.squeeze(inputs['path_to_link'])

		# for i-th path, path_ids indicates the path they are belonging to
		path_ids = tf.squeeze(inputs['path_ids'])
		# for _-th path, sequence_path indicate the number of hoop in each path, max(sequence_path) = max length of q path
		sequence_path = tf.squeeze(inputs['sequence_path'])
		# for _-th link, sequence_link indicate the number of path sharing a link, max (sequence_link) = max number of path sharing a link
		sequence_links = tf.squeeze(inputs['sequence_links'])

		n_links = inputs['n_links']
		n_paths = inputs['n_paths']
		roles = tf.cast(tf.transpose(inputs['role']), tf.float32)

		link_adj = inputs['adj']
		role_adj = inputs['role_adj']
		
		traffic_per_link_sum = inputs ['traffic_per_link_sum']

		# Initialize the initial hidden state fowr links
		traffic_per_link_sum = tf.expand_dims(traffic_per_link_sum, axis=1) # a quick fix, new code
		link_state = tf.concat([(traffic_per_link_sum/2000)/(capacity/100000)], axis=1)
		# import ipdb; ipdb.set_trace()
		link_state = self.link_embedding(link_state)

		# Added code: for DGCN
		adj_1st_prox = inputs ['1st_order_prox']
		adj_2nd_prox_in  = inputs ['2nd_order_prox_in']
		adj_2nd_prox_out = inputs ['2nd_order_prox_out']
		
		adj_1st_prox_edge_index, adj_1st_prox_edge_weight = self.find_adj_edge_index_and_weight(adj_1st_prox)
		adj_2nd_prox_in_edge_index, adj_2nd_prox_in_edge_weight = self.find_adj_edge_index_and_weight(adj_2nd_prox_in)
		adj_2nd_prox_out_edge_index, adj_2nd_prox_out_edge_weight = self.find_adj_edge_index_and_weight(adj_2nd_prox_out)
		

		h_drop_0 = self.dropout(link_state, training=True)
		h_dgcn_0 = self.dgcn0([h_drop_0, 
							adj_1st_prox_edge_index, adj_1st_prox_edge_weight,
							adj_2nd_prox_in_edge_index,adj_2nd_prox_in_edge_weight,
							adj_2nd_prox_out_edge_index,adj_2nd_prox_out_edge_weight
							], training=True)
		
		h_drop_1 = self.dropout(h_dgcn_0, training=True)
		h_dgcn_1 = self.dgcn1([h_drop_1, 
							adj_1st_prox_edge_index, adj_1st_prox_edge_weight,
							adj_2nd_prox_in_edge_index,adj_2nd_prox_in_edge_weight,
							adj_2nd_prox_out_edge_index,adj_2nd_prox_out_edge_weight
							], training=True)

		h_drop_2 = self.dropout(h_dgcn_1, training=True)
		h_dgcn_2 = self.dgcn2([h_drop_2, 
							adj_1st_prox_edge_index, adj_1st_prox_edge_weight,
							adj_2nd_prox_in_edge_index,adj_2nd_prox_in_edge_weight,
							adj_2nd_prox_out_edge_index,adj_2nd_prox_out_edge_weight
							], training=True)


		edge_index_A_role = self.find_role_adj_edge_index(role_adj)
		h_gat_0 = self.gat0([h_drop_0, edge_index_A_role], training=True)

		link_state_large = tf.concat([h_dgcn_0,h_dgcn_1,h_dgcn_2,h_gat_0], axis=1)
		link_state = self.condense_net(link_state_large)

		# Call the readout ANN and return its predictions
		r = self.readout_link(link_state)

		return r

	def find_adj_edge_index_and_weight(self,adj):
		zero = tf.constant(0, dtype=tf.float32)
		where_A_ToF = tf.not_equal(adj, zero)
		where_A_ToF.set_shape([None,None])
		where_A = tf.where(where_A_ToF)
		edge_index = tf.cast(tf.transpose(where_A), tf.int32)
		link_weight = adj[where_A_ToF]	
		if self.config['HYPERPARAMETERS']['dropedge'] == 'Yes':
			edge_index, link_weight = self.dropedge([edge_index, link_weight], training=True)
		return edge_index,link_weight
		
		
	def find_role_adj_edge_index(self,adj):
		zero_int = tf.constant(0, dtype=tf.int64)
		where_A_role_ToF = tf.not_equal(adj, zero_int)
		where_A_role = tf.where(where_A_role_ToF)
		edge_index_A_role = tf.cast(tf.transpose(where_A_role), tf.int32)
		return edge_index_A_role
		