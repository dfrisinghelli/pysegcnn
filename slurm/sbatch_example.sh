#!/bin/bash

# -----------------------------------------------------------------------------
# Usage: sbatch <script.sh> ---------------------------------------------------
# -----------------------------------------------------------------------------

# e.g., $ sbatch sbatch_example.sh

# -----------------------------------------------------------------------------
# Command line options for SLURM sbatch ---------------------------------------
# -----------------------------------------------------------------------------

#SBATCH --job-name=training-pysegcnn

#SBATCH --output=training-pysegcnn-%j.out

#SBATCH --error=training-pysegcnn-%j.err

#SBATCH --partition gpu

#SBATCH --nodes 1

#SBATCH --mem=64G

#SBATCH --gres=gpu:1

#SBATCH --ntasks-per-node=1

# -----------------------------------------------------------------------------
# Commands to execute on requested resources ----------------------------------
# -----------------------------------------------------------------------------

# load conda command line tool
source /home/clusterusers/anaconda3/etc/profile.d/conda.sh

# activate conda environment
conda activate pysegcnn

# change to directory containing script to run
cd ~/git/pysegcnn/pysegcnn/main

# execute script
time python train_source.py
