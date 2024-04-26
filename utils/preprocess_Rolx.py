import os
import configparser
import multiprocessing
import argparse
import pickle

import numpy as np
from numpy.lib.utils import source
import tensorflow as tf
from tqdm import tqdm

from read_dataset import input_fn
from datanetAPI import DatanetAPI
import read_dataset


import networkx as nx
import numpy as np

# Read the config file
config = configparser.ConfigParser()
config._interpolation = configparser.ExtendedInterpolation()

class BatchPreprocessor:
    def __init__(self, source_dir, destination_dir):
        self.source_dir = source_dir
        self.destination_dir = destination_dir

    def proccess_batch(self, batch):
        dnapi = DatanetAPI(self.source_dir)
        for file_tuple in batch:
            
            dnapi.set_files_to_process([file_tuple])
            
            destination_base = file_tuple[0].replace(self.source_dir, self.destination_dir, 1)
            os.makedirs(destination_base, exist_ok=True)
            for idx, sample in enumerate(read_dataset.generator(datanet_obj=dnapi, complete_info=True)):
                with open("{}_{}.pickle".format(os.path.join(destination_base, file_tuple[1]), idx), 'wb') as outf:
                    pickle.dump(sample, outf)
                    


#%%
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--n-threads', required=True, type=int,
        help='Number of threads to use')
    parser.add_argument('-c', '--config', default='./config.ini',
        help='Path to the config.ini file')
    args = parser.parse_args()
    config.read(args.config)

    if False:
        #%%
        ds_test = input_fn(config['DIRECTORIES']['test'], shuffle=False, complete_info=True)
        ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)
        tf.data.experimental.save(ds_test, config['DIRECTORIES']['test'] + '/pre-parsed_2.dataset')

#%%
    n_threads = args.n_threads
    for source_dir, destination_dir in [
            (config['DIRECTORIES']['test'], config['DIRECTORIES']['preprocessed_test']),
            (config['DIRECTORIES']['train'], config['DIRECTORIES']['preprocessed_train'])]:
        dnapi = DatanetAPI(source_dir)
        source_files = np.array(dnapi.get_available_files())
        np.random.shuffle(source_files)  # to make threads complete at similar times
        source_file_batches = np.array_split(source_files, n_threads)
        print("'Preprocessing {} files".format(len(source_files)))
        batch_preprocessor = BatchPreprocessor(source_dir, destination_dir)
        print ('Good')
        with multiprocessing.Pool(n_threads) as p:
            [_ for _ in tqdm(p.imap(batch_preprocessor.proccess_batch, source_file_batches))]
        print ("Nice")
