# %%
from IPython import get_ipython
if get_ipython() is not None:
    get_ipython().run_line_magic('reload_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')
import os

# Uncomment this line in case you want to disable GPU execution
# Note you need to have CUDA installed to run de execution in GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import numpy as np
from glob import iglob
import configparser
from itertools import zip_longest
import zipfile
import pandas as pd
import pickle

import read_dataset
from read_dataset import input_fn
from routenet_model import RouteNetModel

# %%

# Read the config.ini file
config = configparser.ConfigParser()
config._interpolation = configparser.ExtendedInterpolation()
config.read('/proj/gnnet/epedbat_vscode/yifei_latest_model/config.ini')

# %%
# Ensure that directories are loaded in a given order. It is IMPORTANT to keep this, as it ensures that samples
# are loaded in the desired order
directories = [d for d in iglob(config['DIRECTORIES']['test'] + '/*/*')]
# First, sort by scenario and second, by topology size
directories.sort(key=lambda f: (os.path.dirname(f), int(os.path.basename(f))))
directories
# %%
test_data_generator = read_dataset.PreprocessedGenerator(
            data_dir=config['DIRECTORIES']['preprocessed_test'],
            shuffle=False,
            complete_info=True)
list(test_data_generator.get_available_sample_files()[-10:])

# %%
with open('/home/nonroot/persistent-home/yifei_latest_model/results.pickle', 'rb') as inpf:
    df = pickle.load(inpf)
df['batch_idx'] = df['batch_idx'].apply(pd.to_numeric)
df = df.sort_values('batch_idx')
df['scenario'] = df['sample_file'].apply(lambda x: os.path.normpath(x).split(os.path.sep)[-3])
df['topology_size'] = df['sample_file'].apply(lambda x: os.path.normpath(x).split(os.path.sep)[-2])
df['file_name'] = df['sample_file'].apply(lambda x: os.path.normpath(x).split(os.path.sep)[-1])
#df = df.sort_values(['scenario', 'topology_size', 'file_name'])

with open('/home/nonroot/persistent-home/yifei_latest_model/output.tt', 'w') as outf:
    first = True
    for path_occupancy in df.sort_values('batch_idx')['path_delay']:
        if not first:
            outf.write("\n")
        outf.write("{}".format(';'.join([format(i, '.6f') for i in np.squeeze(path_occupancy)])))
        first = False

# %%
upload_file = open(FILENAME+'.txt', "w")

predictions = []
first = True
print('Starting predictions...')
for d in directories:
    print('Current directory: ' + d)

    # It is NECESSARY to keep shuffle as 'False', as samples have to be read always in the same order
    ds_test = input_fn(d, shuffle=False)
    ds_test = ds_test.map(lambda x, y: transformation(x, y))
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)
    ds_test.get_available_files

    # Generate predictions
    pred = model.predict(ds_test)

    # If you need to denormalize or process the model predictions do it here
    # E.g.:
    # y = np.exp(pred)

    # Separate predictions of each sample; each line contains all the per-path predictions of that sample
    # excluding those paths with no traffic (i.e., flow['AvgBw'] != 0 and flow['PktsGen'] != 0)
    idx = 0
    for x, y in ds_test:
        top_pred = pred[idx: idx+int(x['n_paths'])]
        idx += int(x['n_paths'])
        if not first:
            upload_file.write("\n")
        upload_file.write("{}".format(';'.join([format(i,'.6f') for i in np.squeeze(top_pred)])))
        first = False

upload_file.close()

zipfile.ZipFile(FILENAME+'.zip', mode='w').write(FILENAME+'.txt')

########################################################
###### CHECKING THE FORMAT OF THE SUBMISSION FILE ######
########################################################
sample_num = 0
error = False
print("Checking the file...")

with open(FILENAME + '.txt', "r") as uploaded_file, open(PATHS_PER_SAMPLE, "r") as path_per_sample:
    # Load all files line by line (not at once)
    for prediction, n_paths in zip_longest(uploaded_file, path_per_sample):
        # Case 1: Line Count does not match.
        if n_paths is None or prediction is None:
            print("WARNING: File must contain 1560 lines in total for the final test datset (90 for the toy dataset). "
                  "Looks like the uploaded file has {} lines".format(sample_num))
            error = True

        # Remove the \n at the end of lines
        prediction = prediction.rstrip()
        n_paths = n_paths.rstrip()

        # Split the line, convert to float and then, to list
        prediction = list(map(float, prediction.split(";")))

        # Case 2: Wrong number of predictions in a sample
        if len(prediction) != int(n_paths):
            print("WARNING in line {}: This sample should have {} path delay predictions, "
                  "but it has {} predictions".format(sample_num, n_paths, len(prediction)))
            error = True

        sample_num += 1

if not error:
    print("Congratulations! The submission file has passed all the tests! "
          "You can now submit it to the evaluation platform")
