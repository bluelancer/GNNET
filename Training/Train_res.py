Benchmark = False 
scalability_test  = False
New_training = True
ES = False
new_dataset = True
dataset_renorm_by_capacity = True
# dropmore = False
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import tensorflow as tf
tf.config.run_functions_eagerly(True)
import sys
import numpy as np
from utils.read_dataset_role import input_fn
import inspect
import argparse


import configparser
import datetime
from utils.extra_eval import ExtraValidation
from utils.Transform import transformation, transformation_test #, transformation_pretrain


# Read the config file
config = configparser.ConfigParser()
config._interpolation = configparser.ExtendedInterpolation()
config.read('config_res.ini')

###################
### Load model ####
###################

	
if Benchmark:
	from routenet_model import RouteNetModel
elif int(config['HYPERPARAMETERS']['gcn_layers']) == 2:
	print('Note: using 2 layer GCN with res link')
	from RouteRolxNet.routeNALU_GATGCNResDrop_CapacityNORM import RouteNetModel
elif int(config['HYPERPARAMETERS']['gcn_layers']) == 3:
	if config['HYPERPARAMETERS']['directed_graph'] == 'Yes':
		directed = True
		print('Note: using 3 layer Directed-GCN with res link')
		if config['HYPERPARAMETERS']['READOUT'] == 'nalu':
			# if dropmore:
				# print ('droppout extra')
			from RouteRolxNet.routeNALU_GATDGCNResResDrop_CapacityNORM_dropmore import RouteNetModel
			# else:
			# from RouteRolxNet.routeNALU_GATDGCNResResDrop_CapacityNORM import RouteNetModel
		else:
			print('Note: using MLP')
			from RouteRolxNet.routenet_GATDGCNResResDrop_CapacityNORM import RouteNetModel
	else:
		print('Note: using 3 layer GCN with res link')
		if config['HYPERPARAMETERS']['GCN'] == 'gcn':
			print('Note: using GCN with res link')
			if config['HYPERPARAMETERS']['READOUT'] == 'nalu':
				from RouteRolxNet.routeNALU_GATGCNResResDrop_CapacityNORM import RouteNetModel
			else:
				print('Note: using MLP')
				from RouteRolxNet.routenet_GATGCNResResDrop_CapacityNORM import RouteNetModel
		else:
			print('Note: using SGC with res link')
			from RouteRolxNet.routeNALU_GATDSGCResResDrop_CapacityNORM  import RouteNetModel
elif int(config['HYPERPARAMETERS']['gcn_layers']) == 4:
	if config['HYPERPARAMETERS']['use_route'] == 'Yes':
		print('Note: using 4 layer GCN with res link, followed by RNN')
		print('Note: path embedding')
		from RouteRolxNet.routenetNALU_GATGCNResResResDrop_CapacityNORM import RouteNetModel
	elif config['HYPERPARAMETERS']['directed_graph'] == 'Yes':
		directed = True
		print('Note: using 4 layer Directed-GCN with res link')
		if config['HYPERPARAMETERS']['GCN'] == 'gcn':
			print('Note: using GCN with res link')
			if config['HYPERPARAMETERS']['READOUT'] == 'nalu':
				from RouteRolxNet.routeNALU_GATDGCNResResResDrop_CapacityNORM import RouteNetModel
			else:
				from RouteRolxNet.routenet_GATDGCNResResResDrop_CapacityNORM import RouteNetModel
		else:
			print('Note: using SGC with res link')
			from RouteRolxNet.routeNALU_GATDSGCResResResDrop_CapacityNORM import RouteNetModel
	else:
		print('Note: using 4 layer GCN with res link')
		from RouteRolxNet.routenet_GATGCNResResResDrop_CapacityNORM import RouteNetModel
else:
	print('using NALU benchmark')
	from RouteRolxNet.routeNALU_GATGCN_CapacityNORM import RouteNetModel



model_name = inspect.getfile(RouteNetModel).split('/')[-1]

log_dir="logs/fit/"+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S%MS") +'link_state_dim' + config['HYPERPARAMETERS']['link_state_dim'] + 'attention_head = ' + config['HYPERPARAMETERS']['attention_heads']  + 'attetion_units = '+ config['HYPERPARAMETERS']['attention_units']  +  'lr = '+  config['HYPERPARAMETERS']['learning_rate_full'] + "epoch = " + config['RUN_CONFIG']['epochs_full'] +  ' model = ' + model_name

###########################
##### Prepare Dataset #####
###########################

### Initialize the datasets
if new_dataset:
	ds_train = input_fn(config['DIRECTORIES']['new_train'], shuffle=True, pre_processed=True, complete_info = True)
	ds_train = ds_train.map(lambda x, y: transformation_test(x, y,normalize_edge_weight_by_origin_capacity=dataset_renorm_by_capacity))
	ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)
	ds_train = ds_train.repeat()

	ds_test = input_fn(config['DIRECTORIES']['new_test'], shuffle=True, pre_processed=True, complete_info = True)
	ds_test = ds_test.map(lambda x, y: transformation_test(x, y, normalize_edge_weight_by_origin_capacity =dataset_renorm_by_capacity))
	ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)
	
else:
	ds_train = input_fn(config['DIRECTORIES']['preprocessed_train'], shuffle=True)
	ds_train = ds_train.map(lambda x, y,z: transformation(x, y,z))
	ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)
	ds_train = ds_train.repeat()

	ds_test = input_fn(config['DIRECTORIES']['preprocessed_test'], shuffle=False)
	ds_test = ds_test.map(lambda x, y,z: transformation(x, y,z))
	ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

if scalability_test:
	ds_test_train = input_fn(config['DIRECTORIES']['preprocessed_train'], shuffle=False,filter_size = True, filter_operator = "> 49", debug = False)
	ds_test_train = ds_test_train.map(lambda x, y,z: transformation(x, y,z))
	ds_test_train = ds_test_train.prefetch(tf.data.experimental.AUTOTUNE)

	ds_test_same = input_fn(config['DIRECTORIES']['preprocessed_test'], shuffle=True,filter_size = True, filter_operator = "== 50", debug = False)
	ds_test_same = ds_test_same.map(lambda x, y,z: transformation(x, y,z))
	ds_test_same = ds_test_same.prefetch(tf.data.experimental.AUTOTUNE)

	ds_test_large = input_fn(config['DIRECTORIES']['preprocessed_test'], shuffle=True,filter_size = True, filter_operator = "== 120", debug = False)
	ds_test_large = ds_test_large.map(lambda x, y,z: transformation(x, y,z))
	ds_test_large = ds_test_large.prefetch(tf.data.experimental.AUTOTUNE)

	ds_test_larger = input_fn(config['DIRECTORIES']['preprocessed_test'], shuffle=True,filter_size = True, filter_operator = "== 200", debug = False)
	ds_test_larger = ds_test_larger.map(lambda x, y,z: transformation(x, y,z))
	ds_test_larger = ds_test_larger.prefetch(tf.data.experimental.AUTOTUNE)

	ds_test_largest = input_fn(config['DIRECTORIES']['preprocessed_test'], shuffle=True,filter_size = True, filter_operator = "== 260", debug = False)
	ds_test_largest = ds_test_largest.map(lambda x, y,z: transformation(x, y,z))
	ds_test_largest = ds_test_largest.prefetch(tf.data.experimental.AUTOTUNE)

	ds_test_ex = input_fn(config['DIRECTORIES']['preprocessed_test'], shuffle=True,filter_size = True, filter_operator = "== 300", debug = False)
	ds_test_ex = ds_test_ex.map(lambda x, y,z: transformation(x, y,z))
	ds_test_ex = ds_test_ex.prefetch(tf.data.experimental.AUTOTUNE)

	ds_test_train_callback = ExtraValidation(ds_test_train,"{}/{}".format(log_dir,'ds_test_train_callback'),validation_steps =int(config['RUN_CONFIG']['validation_steps']))
	ds_test_same_callback = ExtraValidation(ds_test_same,"{}/{}".format(log_dir,'ds_test_same_callback'),validation_steps =int(config['RUN_CONFIG']['validation_steps']))
	ds_test_large_callback = ExtraValidation(ds_test_large,"{}/{}".format(log_dir,'ds_test_==120_callback'),validation_steps =int(config['RUN_CONFIG']['validation_steps']))
	ds_test_larger_callback = ExtraValidation(ds_test_larger,"{}/{}".format(log_dir,'ds_test_==200_callback'),validation_steps =int(config['RUN_CONFIG']['validation_steps']) )
	ds_test_largest_callback = ExtraValidation(ds_test_largest,"{}/{}".format(log_dir,'ds_test_==260_callback'),validation_steps =int(config['RUN_CONFIG']['validation_steps']) )
	ds_test_ex_callback = ExtraValidation(ds_test_ex,"{}/{}".format(log_dir,'ds_test_==300_callback'),validation_steps =int(config['RUN_CONFIG']['validation_steps']) )


##############################
##### Training  Schedule #####
##############################

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(float(config['HYPERPARAMETERS']['learning_rate_full']),decay_steps=5000,decay_rate=0.96,
	staircase=True)


optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)


#with strategy.scope():
	# Define, build and compile the model
model = RouteNetModel(config)

loss_object = tf.keras.losses.MeanAbsolutePercentageError()

model.compile(loss=loss_object,
		   optimizer=optimizer,
		   run_eagerly=False,
		   metrics="MAPE")

# Define the checkpoint directory where the model will be saved
if New_training:
	ckpt_dir = config['DIRECTORIES']['logs'] +'/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '/' + model_name 
else: 
	ckpt_dir = config['DIRECTORIES']['logs'] +'/' + model_name

latest = tf.train.latest_checkpoint(ckpt_dir)

# Reload the pretrained model in case it exists
if latest is not None:
	print("Found a pretrained model, restoring...")
	model.load_weights(latest)
else:
	print("Starting training from scratch...")


#####################
##### Callbacks #####
#####################

filepath = os.path.join(ckpt_dir, model_name +"{epoch:02d}-{val_loss:.2f}-{val_MAPE:.2f}")

# If save_best_only, the program will only save the best model using 'monitor' as metric
cp_callback = tf.keras.callbacks.ModelCheckpoint(
	filepath=filepath,
	verbose=1,
	mode='min',
	monitor='val_MAPE',
	save_best_only=False,
	save_weights_only=True,
	save_freq='epoch')

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1,write_graph=False, write_images = True)
es_callback =tf.keras.callbacks.EarlyStopping(monitor='val_MAPE', patience=80, restore_best_weights = True)
if not Benchmark: 
	if scalability_test:
		callback_list = [cp_callback,tensorboard_callback,ds_test_train_callback,ds_test_same_callback,ds_test_large_callback,ds_test_larger_callback, ds_test_largest_callback,ds_test_ex_callback]
	else:
		callback_list = [cp_callback,tensorboard_callback]
else: 
	callback_list= [cp_callback,tensorboard_callback,ds_test_train_callback,ds_test_same_callback,ds_test_large_callback,ds_test_larger_callback,ds_test_largest_callback,ds_test_ex_callback]

if ES:
	callback_list.append(es_callback)

#####################
##### Trainings #####
#####################	
	
# This method trains the model saving the model each epoch.
model.fit(ds_train,
		  epochs=int(config['RUN_CONFIG']['epochs_full']),
		  steps_per_epoch=int(config['RUN_CONFIG']['steps_per_epoch_full']),
		  validation_data=ds_test,
		  validation_steps=int(config['RUN_CONFIG']['validation_steps']),
		  callbacks=callback_list,
		  # callbacks=[cp_callback,tensorboard_callback],
		  use_multiprocessing=True)

# This method evaluates the trained model and outputs the desired metrics for all the test dataset.
model.evaluate(ds_test,return_dict=True)

# This method return the predictions in a python array
# predictions = model.predict(ds_test)

# Do stuff here
#print(predictions)
