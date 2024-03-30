#!/bin/bash

#SBATCH -c 10
#SBATCH -n 1   
#SBATCH -N 1
#SBATCH -t 1-00:00:00
#SBATCH --mem=40G
#SBATCH -G a100:1
#SBATCH -p general  
#SBATCH -q public
#SBATCH --mail-type=ALL

module load mamba/latest

source activate inr2array

python -m experiments.attack_tuning --dataset_name MNIST --rundir /scratch/sbajjur3/INR2ARRAY/outputs/2024-01-05/13-57-00 --embedding_path /scratch/sbajjur3/INR2ARRAY/experiments/data/mnist-embeddings.pt --model_path /scratch/sbajjur3/INR2ARRAY/outputs/2024-01-07/19-15-24/best.pt