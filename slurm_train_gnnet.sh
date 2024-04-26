#!/bin/bash
#SBATCH -J train_gnnet
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 8:59:59
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yifeij@kth.se
#SBATCH -o ./slurm_train_gnnet.out
#SBATCH -e ./slurm_train_gnnet.err

micromamba activate /proj/raygnn/workspace/sionna_mamba
cd /proj/raygnn_storage/CELlularOtaGnn/GNNET/Training
python Train_res.py

