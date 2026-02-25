#!/bin/bash
#SBATCH --job-name=cat-dog-logistic-regression
#SBATCH -t 08:00:00
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

source /home/hice1/madewolu9/scratch/madewolu9/LRDVCC/mEoW-oR-wOoF/.venv/bin/activate

python train.py