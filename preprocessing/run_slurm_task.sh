#!/bin/bash
#SBATCH --job-name=gnnet_preprocess       # Set a name for your job
#SBATCH --nodes=1               # Request 1 node
#SBATCH --cpus-per-task=32       # Number of CPU cores per task
#SBATCH --mem=96G                # Request 8 GB of memory
#SBATCH --time=1-23:00:00         # Set the maximum walltime for your job (hh:mm:ss)
#SBATCH --output=preprocessing_output.log     # Define the name of the output log file
#SBATCH --error=preprocessing_error.log       # Define the name of the error log file

# Activate the virtual environment
source /proj/raygnn/workspace/sionna_mamba/bin/activate

# Navigate to the directory where your script is located
cd /proj/raygnn/workspace/GNNET/preprocessing

# Run your script
python preprocess_RolX.py -n 32 -c config.ini

# Deactivate the virtual environment
deactivate
