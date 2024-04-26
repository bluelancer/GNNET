import os

import tensorflow as tf
import numpy as np

# This should be used for benchmark
def transformation_raw(x, y, z):
	return x, x['delay']

def transformation(x, y, z):
	def gen_adj_node_list(node_list_raw, adj):
		nodelist = []
		decoder = np.vectorize(lambda x: x.decode('UTF-8'))
		node_list_decode = decoder(node_list_raw)
		for node in node_list_decode[0]:
			i  = node.replace('l_', '')
			nodelist.append(int(i))

		k = max(nodelist) 
		indice_list = []

		for i in range(k+1):
			indice_list.append(nodelist.index(i))
		b = tf.gather(adj[0],indice_list,axis=0)
		b = tf.gather(b,indice_list,axis=1)

		return indice_list, b

	def tf_parsing_tensor_to_np(tensor):
		return tensor.numpy()

	@tf.function
	def get_role_per_path(roles,link_to_path,path_ids, sequence_path,n_paths):
		# Compute role of each link, consisting a path  
		role_per_link_id = tf.gather(roles, link_to_path)
		role_per_link_id = role_per_link_id + 1 

		max_len_path = tf.reduce_max(sequence_path) + 1
		shape_per_link_path_role= tf.stack([n_paths,max_len_path])
		ids = tf.stack([path_ids, sequence_path], axis=1)
		role_per_path = tf.scatter_nd(ids, role_per_link_id, shape_per_link_path_role)
		link_per_path = tf.scatter_nd(ids, link_to_path+1, shape_per_link_path_role)

		# Role_per_path[Path_id] = role
		# if role>=0 get role, if role < 0 get <end>
		role_per_path = role_per_path -1
		link_per_path = link_per_path -1
		return role_per_path,link_per_path,max_len_path

	def get_link_per_path(path_ids,sequence_path,n_paths,link_to_path):
		ids_links = tf.stack([path_ids, sequence_path], axis=1)
		max_len_path_for_link = tf.reduce_max(sequence_path) + 1
		#tf.print ('max_len_path_for_link',max_len_path_for_link)

		shape_per_link_path_links= tf.stack([n_paths,max_len_path_for_link])
		link_per_path = tf.scatter_nd(ids_links, link_to_path+1, shape_per_link_path_links)
		link_per_path = link_per_path -1
		return link_per_path

	def get_sharing_path_per_link(sequence_link,sequence_path,path_to_link,n_links):
		### find adjancent paths：  adjancent[path] adjacent path, with self loop
		# 1, find per link sharing path

		# a one-tensor that can accumulate link apprearance in sequence_link
		a = tf.ones(tf.shape(sequence_link))
		link_appearance_counter = tf.math.unsorted_segment_sum(a, sequence_link, n_links, name=None)
		max_path_sharing_link =  tf.reduce_max(link_appearance_counter)

		# build path adjancy matix
		shape_per_link_using_paths = tf.stack([int(n_links),int(max_path_sharing_link)])

		# Extend the link_appearance_counter into a sequnce of [0,1,2..appreance Nr of link#1...]
		sequence = tf.zeros([0])
		# for i in link_appearance_counter:
		#    sequence = tf.concat([sequence,tf.range(i,dtype=tf.float32)],axis = 0)
		i = tf.constant(0)
		while_condition = lambda i, sequence: tf.less(i, tf.shape(link_appearance_counter)[0])

		#def body(i,sequence):
		#    # do something here which you want to do in your loop
		#    # increment i
		#    sequence = tf.concat([sequence,tf.range(link_appearance_counter[i],dtype=tf.float32)],axis = 0)
		#    return [tf.add(i, 1)]

		body = lambda i, sequence: (i+1,  tf.concat([sequence,tf.range(link_appearance_counter[i],dtype=tf.float32)],axis = 0))
		i, sequence = tf.while_loop(while_condition, body, [i,sequence] ,shape_invariants=[i.get_shape(),tf.TensorShape([None,])])

		sequence = tf.cast(sequence, tf.int32)
		ids_path_sharing_link = tf.stack([sequence_link, sequence], axis=1)
		path_to_link = path_to_link +1 
		# 2, find per link sharing path
		# path_per_link[link_id] = [sharing path_id,..]
		path_per_link= tf.scatter_nd(ids_path_sharing_link, path_to_link, shape_per_link_using_paths)
		return path_per_link -1, max_path_sharing_link,sequence,link_appearance_counter



	# Step 1: get Adj matrix in correct order
	node_list = tf.py_function(func=tf_parsing_tensor_to_np,inp=[x['node_list']], Tout=[tf.string])
	adj =tf.py_function(func=tf_parsing_tensor_to_np,inp=[x['edge_adj']], Tout=[tf.float64])
	x['node_list'],x['adj']  = tf.py_function(func=gen_adj_node_list, inp=[node_list, adj], Tout=[tf.int64,tf.float64])

	# Step 2: get Role adj matrix

	# 2.1 Get needed data:
	n_paths = x['n_paths']
	n_links =x['n_links']
	path_to_link = x["path_to_link"]
	link_to_path = x['link_to_path']
	sequence_path = x['sequence_path']
	sequence_link = x['sequence_links']
	roles = x["role"]
	path_ids = x['path_ids']

	# 2.2 In Every path, get per-link role:
	# Input: roles,link_to_path,path_ids, sequence_path,n_paths
	# Output role_per_path[Nr_path]= [link_0_role, link_1_role... ]
	role_per_path,link_per_path,max_len_path = get_role_per_path(roles,link_to_path,path_ids, sequence_path,n_paths)

	# 2.3 In Every link, find all sharing path:
	path_per_link,max_path_sharing_link,sequence_path,link_appearance_counter = get_sharing_path_per_link(sequence_link,sequence_path,path_to_link,n_links)
	x ['link_shared_by_n_paths'] = link_appearance_counter
	x ['max_len_path'] = max_len_path
	x ['most_shared_link'] = max_path_sharing_link
	x ['path_per_link'] = path_per_link
	x ['link_per_path'] = link_per_path
	zero_traffic = tf.concat([tf.expand_dims(tf.cast(tf.constant(0),tf.float32),axis =0),x['traffic']],axis = 0)
	traffic_perlink = tf.gather(zero_traffic,path_per_link + 1)
	traffic_per_link_sum = tf.math.reduce_sum(traffic_perlink,axis = 1)
	x ['traffic_per_link_sum'] = traffic_per_link_sum

	# 2,4 get consisting link per path
	#link_per_path = get_link_per_path(path_ids,sequence_path,n_paths,link_to_path)

	# 2.5 get per links sharing path's including link's role sequence:
	# 2.5.1 build a header to handle the '-1', when we push -1 to 0 it will take header's content

	header = tf.expand_dims(-tf.ones(max_len_path, dtype =tf.int32),axis  =0)

	#tf.print('role_per_path',tf.shape(role_per_path))
	#tf.print('link_per_path',tf.shape(link_per_path))

	# 2.5.2 append header on 1st role of role_per_path And link_per_path, apply +1 on indice (link_wrt_sharing_path)
	header_w_role_per_path =  tf.concat([header,role_per_path],axis = 0)
	link_per_sharing_path_consiting_link_role = tf.gather(header_w_role_per_path,path_per_link+1) 
	header_w_link_per_path =  tf.concat([header,link_per_path],axis = 0)
	link_per_sharing_path_consiting_link_num = tf.gather(header_w_link_per_path,path_per_link+1)
	# Get [link * sharing_path_num * each path consisting link] tensors, last dim can be link number (link_per_sharing_path_consiting_link_num) 
	# or role class (link_per_sharing_path_consiting_link_role) respectively

	# 2.6 Get role adjancey matrix [n_link * n_link]
	# Prepare roles as the dim of link_per_sharing_path_consiting_link_role, and find same role between 2 tensors
	roles_dim_aliged_for_equal =  tf.expand_dims(tf.expand_dims(roles,axis = 1),axis = 2)

	# we got this role_adjancey_raw
	role_adjancey_raw = tf.math.equal(roles_dim_aliged_for_equal,link_per_sharing_path_consiting_link_role)
	# according to the role_adjancey_raw which have the role info, we search within the same shape, but in tensor holding link_id
	role_adjancey_minus_one = tf.where(role_adjancey_raw,link_per_sharing_path_consiting_link_num,tf.constant(-1))
	# we reshape the found adjacncy link_id, 
	# from shape [N_link * sharing path * consisiting link number] to [N_link * all possible role adjacent link number]
	adj_next = tf.reshape(role_adjancey_minus_one, [int(n_links),-1])

	# get the axis of the link_id, where the adjancy happens:
	minus_one = tf.constant(-1)
	not_minus_one=tf.not_equal(adj_next, minus_one)
	# if link_i and link_j is adjancent, then, we must can find:
	# adj_next[i] includes value j and j != -1
	adj_id = tf.stack([tf.where(not_minus_one)[:,0],tf.cast(adj_next[not_minus_one], tf.int64)],axis =1)
	# adj_id includes a sequence of role adjancies, where starts from i, including all j that have adjancy with i, then move to i +1
	# Note that j is not acsending order

	# 2.7 Post processing
	# 2.7.1 tf.unique the same adjancey
	adj_unique =  tf.raw_ops.UniqueV2(x = adj_id,axis = tf.zeros(1,dtype= tf.int64)).y
	# role adjancey shape
	shape_role_adj = tf.cast(tf.stack([n_links,n_links]), tf.int64)
	# Adjancy indicator
	one = tf.ones(tf.shape(adj_unique)[0],dtype= tf.int64)
	role_adj = tf.scatter_nd(adj_unique,one,shape_role_adj)

	# 2.7.2 remove self loop
	role_adj = role_adj -tf.linalg.diag(tf.ones(n_links,dtype= tf.int64))

	x['role_adj'] = role_adj
    
	# Todo: Apply clipping
	# y = (np.clip(y.0.03.0.1) -0.03)/0.07
	
	return x, y

def transformation_test(x, y, directed = True, normalize_edge_weight_by_origin_capacity = True):
	
	def gen_adj_node_list(node_list_raw, adj):
		nodelist = []
		decoder = np.vectorize(lambda x: x.decode('UTF-8'))
		node_list_decode = decoder(node_list_raw)
		for node in node_list_decode[0]:
			i  = node.replace('l_', '')
			nodelist.append(int(i))

		k = max(nodelist) 
		indice_list = []

		for i in range(k+1):
			indice_list.append(nodelist.index(i))
		b = tf.gather(adj[0],indice_list,axis=0)
		b = tf.gather(b,indice_list,axis=1)

		return indice_list, b

	def tf_parsing_tensor_to_np(tensor):
		return tensor.numpy()

	@tf.function
	def get_role_per_path(roles,link_to_path,path_ids, sequence_path,n_paths):
		# Compute role of each link, consisting a path  
		role_per_link_id = tf.gather(roles, link_to_path)
		role_per_link_id = role_per_link_id + 1 

		max_len_path = tf.reduce_max(sequence_path) + 1
		shape_per_link_path_role= tf.stack([n_paths,max_len_path])
		ids = tf.stack([path_ids, sequence_path], axis=1)
		role_per_path = tf.scatter_nd(ids, role_per_link_id, shape_per_link_path_role)
		link_per_path = tf.scatter_nd(ids, link_to_path+1, shape_per_link_path_role)

		# Role_per_path[Path_id] = role
		# if role>=0 get role, if role < 0 get <end>
		role_per_path = role_per_path -1
		link_per_path = link_per_path -1
		return role_per_path,link_per_path,max_len_path

	def get_link_per_path(path_ids,sequence_path,n_paths,link_to_path):
		ids_links = tf.stack([path_ids, sequence_path], axis=1)
		max_len_path_for_link = tf.reduce_max(sequence_path) + 1
		#tf.print ('max_len_path_for_link',max_len_path_for_link)

		shape_per_link_path_links= tf.stack([n_paths,max_len_path_for_link])
		link_per_path = tf.scatter_nd(ids_links, link_to_path+1, shape_per_link_path_links)
		link_per_path = link_per_path -1
		return link_per_path

	def get_sharing_path_per_link(sequence_link,sequence_path,path_to_link,n_links):
		### find adjancent paths：  adjancent[path] adjacent path, with self loop
		# 1, find per link sharing path

		# a one-tensor that can accumulate link apprearance in sequence_link
		a = tf.ones(tf.shape(sequence_link))
		link_appearance_counter = tf.math.unsorted_segment_sum(a, sequence_link, n_links, name=None)
		max_path_sharing_link =  tf.reduce_max(link_appearance_counter)

		# build path adjancy matix
		shape_per_link_using_paths = tf.stack([int(n_links),int(max_path_sharing_link)])

		# Extend the link_appearance_counter into a sequnce of [0,1,2..appreance Nr of link#1...]
		sequence = tf.zeros([0])
		# for i in link_appearance_counter:
		#    sequence = tf.concat([sequence,tf.range(i,dtype=tf.float32)],axis = 0)
		i = tf.constant(0)
		while_condition = lambda i, sequence: tf.less(i, tf.shape(link_appearance_counter)[0])

		#def body(i,sequence):
		#    # do something here which you want to do in your loop
		#    # increment i
		#    sequence = tf.concat([sequence,tf.range(link_appearance_counter[i],dtype=tf.float32)],axis = 0)
		#    return [tf.add(i, 1)]

		body = lambda i, sequence: (i+1,  tf.concat([sequence,tf.range(link_appearance_counter[i],dtype=tf.float32)],axis = 0))
		i, sequence = tf.while_loop(while_condition, body, [i,sequence] ,shape_invariants=[i.get_shape(),tf.TensorShape([None,])])

		sequence = tf.cast(sequence, tf.int32)
		ids_path_sharing_link = tf.stack([sequence_link, sequence], axis=1)
		path_to_link = path_to_link +1 
		# 2, find per link sharing path
		# path_per_link[link_id] = [sharing path_id,..]
		path_per_link= tf.scatter_nd(ids_path_sharing_link, path_to_link, shape_per_link_using_paths)
		return path_per_link -1, max_path_sharing_link,sequence,link_appearance_counter

	def check_symmetric(a, rtol=1e-05, atol=1e-08):
		return np.allclose(a.numpy(), a.numpy().T, rtol=rtol, atol=atol)


	# Step 1: get Adj matrix in correct order
	node_list = tf.py_function(func=tf_parsing_tensor_to_np,inp=[x['node_list']], Tout=[tf.string])
	adj =tf.py_function(func=tf_parsing_tensor_to_np,inp=[x['edge_adj']], Tout=[tf.float64])
	x['node_list'],x['adj']  = tf.py_function(func=gen_adj_node_list, inp=[node_list, adj], Tout=[tf.int64,tf.float64])
		
	
	if normalize_edge_weight_by_origin_capacity:
		capacity = tf.expand_dims(x['capacity'],axis = 0)
		new_adj = tf.cast(x['adj'],tf.float32)*capacity
		new_adj = new_adj/tf.transpose(capacity)
		x['capacity_norm_adj'] =new_adj
		adj = new_adj
	else:
		adj = x['adj']

	adj = tf.cast(adj,tf.float64)	
	# Adding: Compute 1st. 2nd_in and 2nd_out adj matrix
	x['1st_order_prox'] = (adj+tf.transpose(adj))/2
	x['2nd_order_prox_in'] = tf.matmul(tf.transpose(adj) ,adj/tf.expand_dims(tf.reduce_sum(adj,axis = 1)+1e-8,axis = 1))
	x['2nd_order_prox_out'] = tf.matmul(adj,tf.transpose(adj)/tf.expand_dims(tf.reduce_sum(adj,axis = 0)+1e-8,axis = 1))
	
	if directed:
		x['adj'] = (adj+tf.transpose(adj))/2
		
	# Step 2: get Role adj matrix
	# 2.1 Get needed data:
	n_paths = x['n_paths']
	n_links =x['n_links']
	path_to_link = x["path_to_link"]
	link_to_path = x['link_to_path']
	sequence_path = x['sequence_path']
	sequence_link = x['sequence_links']
	roles = x["role"]
	path_ids = x['path_ids']

	# 2.2 In Every path, get per-link role:
	# Input: roles,link_to_path,path_ids, sequence_path,n_paths
	# Output role_per_path[Nr_path]= [link_0_role, link_1_role... ]
	role_per_path,link_per_path,max_len_path = get_role_per_path(roles,link_to_path,path_ids, sequence_path,n_paths)

	# 2.3 In Every link, find all sharing path:
	path_per_link,max_path_sharing_link,sequence_path,link_appearance_counter = get_sharing_path_per_link(sequence_link,sequence_path,path_to_link,n_links)
	x ['link_shared_by_n_paths'] = link_appearance_counter
	x ['max_len_path'] = max_len_path
	x ['most_shared_link'] = max_path_sharing_link
	x ['path_per_link'] = path_per_link
	x ['link_per_path'] = link_per_path
	zero_traffic = tf.concat([tf.expand_dims(tf.cast(tf.constant(0),tf.float32),axis =0),x['traffic']],axis = 0)
	traffic_perlink = tf.gather(zero_traffic,path_per_link + 1)
	traffic_per_link_sum = tf.math.reduce_sum(traffic_perlink,axis = 1)
	x ['traffic_per_link_sum'] = traffic_per_link_sum

	# 2,4 get consisting link per path
	#link_per_path = get_link_per_path(path_ids,sequence_path,n_paths,link_to_path)

	# 2.5 get per links sharing path's including link's role sequence:
	# 2.5.1 build a header to handle the '-1', when we push -1 to 0 it will take header's content

	header = tf.expand_dims(-tf.ones(max_len_path, dtype =tf.int32),axis  =0)

	#tf.print('role_per_path',tf.shape(role_per_path))
	#tf.print('link_per_path',tf.shape(link_per_path))

	# 2.5.2 append header on 1st role of role_per_path And link_per_path, apply +1 on indice (link_wrt_sharing_path)
	header_w_role_per_path =  tf.concat([header,role_per_path],axis = 0)
	link_per_sharing_path_consiting_link_role = tf.gather(header_w_role_per_path,path_per_link+1) 
	header_w_link_per_path =  tf.concat([header,link_per_path],axis = 0)
	link_per_sharing_path_consiting_link_num = tf.gather(header_w_link_per_path,path_per_link+1)
	# Get [link * sharing_path_num * each path consisting link] tensors, last dim can be link number (link_per_sharing_path_consiting_link_num) 
	# or role class (link_per_sharing_path_consiting_link_role) respectively

	# 2.6 Get role adjancey matrix [n_link * n_link]
	# Prepare roles as the dim of link_per_sharing_path_consiting_link_role, and find same role between 2 tensors
	roles_dim_aliged_for_equal =  tf.expand_dims(tf.expand_dims(roles,axis = 1),axis = 2)

	# we got this role_adjancey_raw
	role_adjancey_raw = tf.math.equal(roles_dim_aliged_for_equal,link_per_sharing_path_consiting_link_role)
	# according to the role_adjancey_raw which have the role info, we search within the same shape, but in tensor holding link_id
	role_adjancey_minus_one = tf.where(role_adjancey_raw,link_per_sharing_path_consiting_link_num,tf.constant(-1))
	# we reshape the found adjacncy link_id, 
	# from shape [N_link * sharing path * consisiting link number] to [N_link * all possible role adjacent link number]
	adj_next = tf.reshape(role_adjancey_minus_one, [int(n_links),-1])

	# get the axis of the link_id, where the adjancy happens:
	minus_one = tf.constant(-1)
	not_minus_one=tf.not_equal(adj_next, minus_one)
	# if link_i and link_j is adjancent, then, we must can find:
	# adj_next[i] includes value j and j != -1
	adj_id = tf.stack([tf.where(not_minus_one)[:,0],tf.cast(adj_next[not_minus_one], tf.int64)],axis =1)
	# adj_id includes a sequence of role adjancies, where starts from i, including all j that have adjancy with i, then move to i +1
	# Note that j is not acsending order

	# 2.7 Post processing
	# 2.7.1 tf.unique the same adjancey
	adj_unique =  tf.raw_ops.UniqueV2(x = adj_id,axis = tf.zeros(1,dtype= tf.int64)).y
	# role adjancey shape
	shape_role_adj = tf.cast(tf.stack([n_links,n_links]), tf.int64)
	# Adjancy indicator
	one = tf.ones(tf.shape(adj_unique)[0],dtype= tf.int64)
	role_adj = tf.scatter_nd(adj_unique,one,shape_role_adj)

	# 2.7.2 remove self loop
	role_adj = role_adj -tf.linalg.diag(tf.ones(n_links,dtype= tf.int64))

	x['role_adj'] = role_adj
    
	# Todo: Apply clipping
	# y = (np.clip(y.0.03.0.1) -0.03)/0.07
	
	return x, y

def transformation_bm(x, y):
	return x, x['delay']