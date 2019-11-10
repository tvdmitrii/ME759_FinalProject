#!/usr/bin/env bash
#SBATCH --job-name=Turygin_Project
#SBATCH --sockets-per-node=1 
#SBATCH --partition=slurm_shortgpu
#SBATCH --ntasks=1 --cpus-per-task=1
#SBATCH --time=0-00:00:30

cd $SLURM_SUBMIT_DIR
#./generator 128 8
#./generator 512 8
./generator 700 5
