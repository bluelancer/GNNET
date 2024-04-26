"""
   Copyright 2021 Universitat Politècnica de Catalunya
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
	   http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
import os
import pickle

import numpy as np
import tensorflow as tf
import networkx as nx
from datanetAPI import DatanetAPI, DatanetException
from graphrole import RecursiveFeatureExtractor, RoleExtractor

import sys
import pdb

MODEL_DATA_COLUMNS = [
	"role",
	"edge_adj",
	"node_list", 
	"traffic",
	"packets",
	"capacity",
	"link_to_path",
	"path_to_link",
	"path_ids",
	"sequence_links",
	"sequence_path",
	"n_links",
	"n_paths",
	"global_packet"]

N_ROLES = 5

class PreprocessedGenerator:
	def __init__(self, data_dir=None, shuffle=False, complete_info=False,filter_size = False, filter_operator = None, debug = False):
		self._data_dir = data_dir
		self.complete_info = complete_info
		self._file_paths = None
		self._shuffle = shuffle
		self.filter_size = filter_size
		if self.filter_size:
			self.filter_operator = filter_operator
		self.debug = debug
	
	@property
	def data_dir(self):
		if self._data_dir is not None:
			return self._data_dir
		else:
			raise DatanetException('Object was created without an data_dir')

	def get_available_sample_files(self):
		if self._file_paths is None:
			self._file_paths = []
			for root, path, filenames in os.walk(self.data_dir):
				for filename in filenames:
					# When TF calls it filename comes as bytes :/
					if type(filename) is bytes:
						extension = b'.pickle'
					else:
						extension = '.pickle'
					if filename.endswith(extension):
						if self.filter_size:
							if self.debug:
								print('splitting root',root)
							if eval(os.path.split(root.decode('utf-8'))[1] + self.filter_operator.decode()):
								self._file_paths.append(os.path.join(root, filename))
						else:
							self._file_paths.append(os.path.join(root, filename))
			if self._shuffle:
				self._file_paths = np.array(self._file_paths)
				np.random.shuffle(self._file_paths)
			else:
				# When TF calls it filename comes as bytes :/
				print ('note', self._file_paths)
				if len(self._file_paths) != 0:
					if type(self._file_paths[0]) is bytes:
						self._file_paths = list(map(bytes.decode, self._file_paths))
					# Sort to guarantee that it will always return the same sequence of files
					# First, sort by scenario and second, by topology size
					self._file_paths.sort(
						key=lambda x: (
							os.path.normpath(x).split(os.path.sep)[-3],
							int(os.path.normpath(x).split(os.path.sep)[-2]),
							os.path.normpath(x).split(os.path.sep)[-1]
							))
					self._file_paths = np.array(self._file_paths)
		return self._file_paths

	def set_files_to_process(self, files):
		self._file_paths = np.array(files)
		if self._shuffle:
			np.random.shuffle(self._file_paths)

	def __call__(self):
		for file_path in self.get_available_sample_files():
			with open(file_path, 'rb') as inf:
				input_data = pickle.load(inf)
			if not self.complete_info:
				input_data = ({column: input_data[0][column] for column in MODEL_DATA_COLUMNS}, input_data[1], input_data[2])
			yield input_data

def generator(data_dir=None, shuffle=False, complete_info=False, datanet_obj=None):
	if data_dir is None and datanet_obj is None:
		raise DatanetException("At least one of 'data_dir' or 'datanet_obj' is needed")
	if data_dir:
		tool = DatanetAPI(data_dir.decode('UTF-8'), shuffle=shuffle)
	else:
		tool = datanet_obj
	it = iter(tool)
	num_samples = 0
	for sample in it:
		G_copy = sample.get_topology_object().copy()
		T = sample.get_traffic_matrix()
		R = sample.get_routing_matrix()
		D = sample.get_performance_matrix()
		P = sample.get_port_stats()
		HG = network_to_hypergraph_better(network_graph=G_copy,
								   routing_matrix=R,
								   traffic_matrix=T,
								   performance_matrix=D,
								   port_stats=P)
		num_samples += 1
		input_data = hypergraph_to_input_data_better(HG)
		if not complete_info:
			input_data = ({column: input_data[0][column] for column in MODEL_DATA_COLUMNS}, input_data[1])
		yield input_data


def network_to_hypergraph_better(network_graph, routing_matrix, traffic_matrix, performance_matrix, port_stats):
	G = nx.DiGraph(network_graph)
	R = routing_matrix
	T = traffic_matrix
	D = performance_matrix
	P = port_stats
	traffic_agg_by_hop  = np.zeros((int(G.number_of_nodes()),int(G.number_of_nodes())))
	#traffic_agg  = np.zeros((int(G.number_of_nodes()),int(G.number_of_nodes())))

	D_G = nx.DiGraph()

	for src in range(G.number_of_nodes()):
		for dst in range(G.number_of_nodes()):
			if src != dst:
				if G.has_edge(src, dst):
					D_G.add_node('l_{}_{}'.format(src, dst),
								 capacity=G.edges[src, dst]['bandwidth'],
								 occupancy=P[src][dst]['qosQueuesStats'][0]['avgPortOccupancy'] /
											G.nodes[src]['queueSizes'],
								 average_packet_size=P[src][dst]['qosQueuesStats'][0]['avgPacketSize'],
                                 queue_size_packets=G.nodes[src]['queueSizes'])

				for f_id in range(len(T[src, dst]['Flows'])):
					if T[src, dst]['Flows'][f_id]['AvgBw'] != 0 and T[src, dst]['Flows'][f_id]['PktsGen'] != 0:

						D_G.add_node('p_{}_{}_{}'.format(src, dst, f_id),
									 traffic=T[src, dst]['Flows'][f_id]['AvgBw'],
									 packets=T[src, dst]['Flows'][f_id]['PktsGen'],
									 delay=D[src, dst]['Flows'][f_id]['AvgDelay'])

						for h_1, h_2 in [R[src, dst][i:i + 2] for i in range(0, len(R[src, dst]) - 1)]:
							D_G.add_edge('p_{}_{}_{}'.format(src, dst, f_id), 'l_{}_{}'.format(h_1, h_2), type = 0)
							D_G.add_edge('l_{}_{}'.format(h_1, h_2), 'p_{}_{}_{}'.format(src, dst, f_id), type = 0)
							traffic_agg_by_hop[h_1][h_2]  = traffic_agg_by_hop[h_1][h_2] +  T[src, dst]['Flows'][f_id]['AvgBw']

						if len(R[src, dst]) > 2:
							for h_1, h_2, h_3 in [R[src, dst][i:i + 3] for i in range(0, len(R[src, dst]) - 2)]: 
								 traffic_agg_by_hop[h_1][h_3] = traffic_agg_by_hop[h_1][h_3] +  T[src, dst]['Flows'][f_id]['AvgBw']

	D_G.remove_nodes_from([node for node, out_degree in D_G.out_degree() if out_degree == 0])

	# Refactor 
	for src in range(G.number_of_nodes()):
		for dst in range(G.number_of_nodes()):
			if (src != dst):
				if len(R[src, dst]) == 2 and len(T[src, dst]['Flows']) != 0 and T[src, dst]['Flows'][f_id]['PktsGen'] != 0:
					# if this is a 2 hop routing, it is a self loop
					for h_1, h_2 in [R[src, dst][i:i + 2] for i in range(0, len(R[src, dst]) - 1)]: 
						D_G.add_edge('l_{}_{}'.format(h_1, h_2),'l_{}_{}'.format(h_1, h_2), type = 1, weight = traffic_agg_by_hop[h_1][h_2]/G.edges[h_1, h_2]['bandwidth'])
				if len(R[src, dst]) > 2:
					# if this is more than 3 hop routing:
					for h_1, h_2, h_3 in [R[src, dst][i:i + 3] for i in range(0, len(R[src, dst]) - 2)]:  
						h_1, h_2, h_3 = int(h_1),int(h_2),int(h_3)
						if len(T[src, dst]['Flows']) != 0 and T[src, dst]['Flows'][f_id]['PktsGen'] != 0:
							#common_traffic = traffic_agg[h_1][h_3]/min(G.edges[h_1, h_2]['bandwidth'],G.edges[h_2, h_3]['bandwidth'])    
							common_traffic = traffic_agg_by_hop[h_1][h_3]/min(G.edges[h_1, h_2]['bandwidth'],G.edges[h_2, h_3]['bandwidth'])
							fore_traffic =  traffic_agg_by_hop[h_1][h_2]/G.edges[h_1, h_2]['bandwidth']
							after_traffic = traffic_agg_by_hop[h_2][h_3]/G.edges[h_2, h_3]['bandwidth']
							mutual_information = common_traffic/(fore_traffic + after_traffic)
							D_G.add_edge('l_{}_{}'.format(h_1, h_2),'l_{}_{}'.format(h_2, h_3), type = 1, weight = mutual_information)
	return D_G


def hypergraph_to_input_data_better (hypergraph):
	n_p = 0
	n_l = 0
	mapping = {}

	for entity in list(hypergraph.nodes()):
		if entity.startswith('p'):
			mapping[entity] = ('p_{}'.format(n_p))
			n_p += 1
		elif entity.startswith('l'):
			mapping[entity] = ('l_{}'.format(n_l))
			n_l += 1


	D_G = nx.relabel_nodes(hypergraph, mapping)

	#print ('capacity length', len(capacity_dict.keys()))

	link_to_path = []
	path_ids = []
	sequence_path = []

	for i in range(n_p):
		seq_len = 0
		for elem in D_G['p_{}'.format(i)]:
			link_to_path.append(int(elem.replace('l_', '')))
			seq_len += 1
		path_ids.extend(np.full(seq_len, i))
		sequence_path.extend(range(seq_len))

	path_to_link = []
	sequence_links = []
	unique_link_roles = []

	selected_edges = [(u,v,e) for u,v,e in D_G.edges(data=True) if e['type'] == 1]


	G_link = nx.Graph()
	G_link.add_edges_from(selected_edges)  

	feature_extractor = RecursiveFeatureExtractor(G_link)
	features = feature_extractor.extract_features()

	# give 5 roles for fast result
	role_extractor = RoleExtractor(n_roles = N_ROLES)
	role_extractor.extract_role_factors(features)    
	node_roles = role_extractor.roles
	#print ('node_roles',node_roles)

	# Have checked printed roles are same as prev method
	for i in range(n_l):
		seq_len = 0
		node_i_unique_role = node_roles['l_{}'.format(i)]
		node_i_unique_role_int = int(node_i_unique_role.replace('role_', ''))
		unique_link_roles.append(node_i_unique_role_int)
		for elem in D_G['l_{}'.format(i)]:
			if elem.startswith('p'):
				path_to_link.append(int(elem.replace('p_', '')))
				seq_len += 1 
		sequence_links.extend(np.full(seq_len, i))

	#print (len(unique_link_roles), len(list(nx.get_node_attributes(D_G, 'capacity').values())))

	nodes_list = list(G_link.nodes())
	A_link_np = nx.to_numpy_array(G_link, nodelist= nodes_list) 
	D_G_nodes_list = []
	for edge in list(D_G.nodes()):
		if edge.startswith('l'):
			D_G_nodes_list.append(edge)
	capacity_dict = nx.get_node_attributes(D_G, 'capacity')

	# The length of roles should be matching with the length of capacities
	try:
		assert len(unique_link_roles)==len(list(nx.get_node_attributes(D_G, 'capacity').values()))
	except:
		print ('capacity length', len(capacity_dict.keys()),
				'len(nodes_list)',len(nodes_list),
			  'len(selected_edges)', len(selected_edges),
			  'D_G_nodes_list',len(D_G_nodes_list),
			  'n_l',n_l, 
			  "len(capacity)", len(list(nx.get_node_attributes(D_G, 'capacity').values())),
			  "capacity_dict",capacity_dict)        
		capacity_link_list = []
		for entity in D_G_nodes_list:
					try:
						if capacity_dict[entity]:
							pass
						else:
							print ('entity missing capacity', entity)   
					except:
						print ('this entity is not exist in D_G')
						print ('missing entity',entity )      
						print ('avalible entity', [i in D_G['l_{}'.format(i)]])
						pdb.set_trace()


	return  {
			"role": unique_link_roles,
			"edge_adj": A_link_np,
			"node_list": nodes_list,
			"traffic": list(nx.get_node_attributes(D_G, 'traffic').values()),
			"packets": list(nx.get_node_attributes(D_G, 'packets').values()),
			"capacity": list(nx.get_node_attributes(D_G, 'capacity').values()),
			"link_to_path": link_to_path,
			"path_to_link": path_to_link,
			"path_ids": path_ids,
			"sequence_links": sequence_links,
			"sequence_path": sequence_path,
			"n_links": n_l,
			"n_paths": n_p,
            "queue_size_packets": list(nx.get_node_attributes(D_G, 'queue_size_packets').values()),
            "average_packet_size": list(nx.get_node_attributes(D_G, 'average_packet_size').values()),
            "delay": list(nx.get_node_attributes(D_G, 'delay').values())
			}, list(nx.get_node_attributes(D_G, 'occupancy').values())

def input_fn(data_dir, shuffle=False, samples=None, complete_info=False, pre_processed=True, filter_size = False, filter_operator = "False", debug = False):

	feature_types={"role": tf.int32,
			 "edge_adj":  tf.float64,
			"node_list": tf.string,   
			"traffic": tf.float32,
			"packets": tf.float32,
			"capacity": tf.float32,
			"link_to_path": tf.int32,
			"path_to_link": tf.int32, "path_ids": tf.int32,
			"sequence_links": tf.int32, "sequence_path": tf.int32,
			"n_links": tf.int32, "n_paths": tf.int32,'global_packet': tf.float32}

	feature_shapes={"role": tf.TensorShape([None]),
		"edge_adj": tf.TensorShape([None,None]),
		"node_list": tf.TensorShape([None]),       
		"traffic": tf.TensorShape([None]),
		"packets": tf.TensorShape([None]),
		"capacity": tf.TensorShape([None]),
		"link_to_path": tf.TensorShape([None]),
		"path_to_link": tf.TensorShape([None]),
		"path_ids": tf.TensorShape([None]),
		"sequence_links": tf.TensorShape([None]),
		"sequence_path": tf.TensorShape([None]),
		"n_links": tf.TensorShape([]),
		"n_paths": tf.TensorShape([]),
		'global_packet':tf.TensorShape([])}

	if complete_info:
		feature_types['queue_size_packets'] = tf.float32
		feature_types['average_packet_size'] = tf.float32
		feature_types['delay'] = tf.float32
		feature_shapes['queue_size_packets'] = tf.TensorShape([None])
		feature_shapes['average_packet_size'] = tf.TensorShape([None])
		feature_shapes['delay'] = tf.TensorShape([None])
		output_types = (feature_types, tf.float32)
		output_shapes = (feature_shapes, tf.TensorShape([None]))
	else:
		output_types = (feature_types, tf.float32, tf.float32)
		output_shapes = (feature_shapes, tf.TensorShape([None]), tf.TensorShape([None]))

	ds = tf.data.Dataset.from_generator((lambda *x: PreprocessedGenerator(*x)()) if pre_processed else generator,
										args=[data_dir, shuffle, complete_info,filter_size,filter_operator,debug],
										output_types=output_types,
										output_shapes=output_shapes,
										)

	if samples:
		ds = ds.take(samples)

	ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

	return ds
