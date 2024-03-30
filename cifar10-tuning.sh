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

python -m experiments.attack_tuning --dataset_name CIFAR10 --rundir /scratch/sbajjur3/INR2ARRAY/outputs/2024-03-21/22-13-54 --embedding_path /scratch/sbajjur3/INR2ARRAY/experiments/data/cifar10-embeddings.pt --model_path /scratch/sbajjur3/INR2ARRAY/outputs/2024-03-20/17-26-03/best.pt