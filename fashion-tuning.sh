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

python -m experiments.attack_tuning --dataset_name FashionMNIST --rundir /scratch/vkanakav/nfn-main/outputs/2024-03-21/00-16-57 --embedding_path /scratch/vkanakav/nfn-main/experiments/data/fashion-embeddings.pt --model_path /scratch/vkanakav/nfn-main/outputs/2024-03-21/15-04-32/best.pt