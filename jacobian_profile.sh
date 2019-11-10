#!/usr/bin/env bash
#SBATCH --job-name=Project
#SBATCH --sockets-per-node=1 
#SBATCH --partition=slurm_shortgpu
#SBATCH --ntasks=1 --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-00:00:30

cd $SLURM_SUBMIT_DIR

nvprof -o jacobian.nvprof  ./jacobian-debug inputArray1048576-8.inp output1048576-8.txt 0.1 >> jacobian.txt
