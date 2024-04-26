import argparse
import multiprocessing
import pickle
import configparser
import os

import zipfile

from train import get_model
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from read_dataset import input_fn
import read_dataset
from Transform import transformation, transformation_test #, transformation_pretrain
import datetime

PATHS_PER_SAMPLE = './utils/paths_per_sample_test_dataset.txt'

# In case you want to disable GPU execution uncomment this line
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Read the config file
config = configparser.ConfigParser()
config._interpolation = configparser.ExtendedInterpolation()
config.read('./config.ini')
time= datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
dir_result = config['DIRECTORIES']['results_folder'] +time + '/'
os.makedirs(dir_result, exist_ok=True)
print('loading dir', config['DIRECTORIES']['preprocessed_test'])
if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-p', '--predict', action='store_true',
		help='Run the ML model against samples and save the results in files')
	parser.add_argument('-e', '--evaluate', action='store_true',
		help='Use previous ML occupancy predictions to compute delay metrics')
	parser.add_argument('-n', '--n-threads', default=multiprocessing.cpu_count(), type=int,
		help='Number of threads to use for evaluation')
	parser.add_argument('--submission-file-prefix',
		help='Name of the challenge submission file to generate')
	parser.add_argument('--model-weights',
		help='Overwrite the model location')
	args = parser.parse_args()
	print('notice',config['DIRECTORIES']['preprocessed_test'])
	if args.predict:
		ds_test = input_fn(
			config['DIRECTORIES']['preprocessed_test'],
			shuffle=False,
			pre_processed=True,
			complete_info=True)
		ds_test = ds_test.map(lambda x, y: transformation_test(x, y, normalize_edge_weight_by_origin_capacity=True))
		ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)
		model = get_model(args.model_weights)

		class SaveResultCallback(keras.callbacks.Callback):
			def on_predict_batch_end(self, batch, logs=None):
				file_name = os.path.join(
					dir_result,
					'batch_{}.pickle'.format(batch))
				with open(file_name, 'wb') as outf:
					pickle.dump(logs['outputs'], outf)

		test_l_pred = model.predict(
			ds_test,
			callbacks=[SaveResultCallback(), keras.callbacks.ProgbarLogger()],
			)
		pd.DataFrame(test_l_pred).rename({0: 'link_delay'}, axis='columns')\
			.to_feather(os.path.join(dir_result, 'all.feather'))

	if args.evaluate:
		test_data_generator = read_dataset.PreprocessedGenerator(
			data_dir=config['DIRECTORIES']['preprocessed_test'],
			shuffle=False,
			complete_info=True)

		def evaluate_worker(worker_input):
			worker_id, result_to_sample_map = worker_input
			batch_idxs, files_to_process = result_to_sample_map.T
			data_generator = read_dataset.PreprocessedGenerator(shuffle=False, complete_info=True)
			data_generator.set_files_to_process(files_to_process)
			worker_delay_ape = []
			worker_occupancy_ape = []
			worker_occupancy_data = {
				'batch_idx': [],
				'occupancy': [],
				'path_delay': [],
				'sample_file': [],
			}
			for batch_idx, sample_file_name, sample in tqdm(
					zip(batch_idxs, files_to_process, iter(data_generator())),
					position=worker_id,
					total=len(batch_idxs),
					desc='Job {}'.format(worker_id)):
				# sample = transformation_test(sample[0], sample[1], normalize_edge_weight_by_origin_capacity=True)
				g = sample[0]
				true_occupancy = pd.Series(sample[1])
				infname = os.path.join(
						dir_result,
						'batch_{}.pickle'.format(batch_idx))
				with open(infname, 'rb') as inf:
					pred_occupancy = pickle.load(inf)[:, 0]

				true_delay = pd.Series(g['delay'])
				link_to_path = pd.Series(g['link_to_path'])
				capacity = pd.Series(g['capacity'])
				capacity_to_path = capacity[link_to_path]
				queue_size_packets = pd.Series(g['queue_size_packets'])
				# worst case we can use 1000 here are the results are not much worse
				average_packet_size = 1000.036033  # median
				# average_packet_size = pd.Series(g['average_packet_size'])
				queue_size_to_path = pd.Series(queue_size_packets * average_packet_size)[link_to_path]
				true_occupancy_to_path = true_occupancy[link_to_path]
				pred_occupancy_to_path = pred_occupancy[link_to_path]

				link_delay_to_path = pred_occupancy_to_path * (queue_size_to_path / capacity_to_path)
				df = pd.DataFrame({
					'path_id': g['path_ids'],
					'link_delay': link_delay_to_path
				})
				predicted_path_delay = df.groupby('path_id')['link_delay'].sum()
				worker_delay_ape.extend(
					100 * abs(predicted_path_delay - true_delay) / true_delay)
				worker_occupancy_ape.extend(
					100 * abs(pred_occupancy - true_occupancy) / true_occupancy)
				worker_occupancy_data['batch_idx'].append(batch_idx)
				worker_occupancy_data['occupancy'].append(pred_occupancy)
				worker_occupancy_data['path_delay'].append(predicted_path_delay)
				worker_occupancy_data['sample_file'].append(sample_file_name)
			return (worker_delay_ape, worker_occupancy_ape, worker_occupancy_data)

		n_threads = args.n_threads
		all_delay_ape = []
		all_occupancy_ape = []
		files_to_process = test_data_generator.get_available_sample_files()
		input_data = np.array((np.arange(files_to_process.shape[0]), files_to_process)).T
		np.random.shuffle(input_data)
		input_data_batches = enumerate(np.array_split(input_data, n_threads))
		all_occupancy_data = None
		with multiprocessing.Pool(n_threads) as p:
			for delay_ape, occupancy_ape, occupancy_data in \
					p.imap(evaluate_worker, input_data_batches):
				all_delay_ape.extend(delay_ape)
				all_occupancy_ape.extend(occupancy_ape)
				if all_occupancy_data is None:
					all_occupancy_data = pd.DataFrame(occupancy_data)
				else:
					all_occupancy_data = all_occupancy_data.append(pd.DataFrame(occupancy_data))

		if args.submission_file_prefix is not None:
			all_occupancy_data['batch_idx'] = all_occupancy_data['batch_idx'].apply(pd.to_numeric)
			with open(args.submission_file_prefix + time+ '.csv', 'w') as outf:
				first = True
				for path_occupancy in all_occupancy_data.sort_values('batch_idx')['path_delay']:
					if not first:
						outf.write("\n")
					outf.write("{}".format(';'.join([format(i, '.6f') for i in np.squeeze(path_occupancy)])))
					first = False
			with zipfile.ZipFile(
					args.submission_file_prefix+ time + '.zip',
					mode='w',
					compression=zipfile.ZIP_DEFLATED) as zip_out:
				zip_out.write(args.submission_file_prefix + time+ '.csv')


		all_delay_ape = np.array(all_delay_ape)
		all_occupancy_ape = np.array(all_occupancy_ape)
		print('')
		print('Total occupancy samples {:,.6f}'.format(all_occupancy_ape.shape[0]))
		print('Occupancy MAPE {:,.6f}'.format(all_occupancy_ape.mean()))
		print('Delay MAPE {:,.6f}'.format(all_delay_ape.mean()))
