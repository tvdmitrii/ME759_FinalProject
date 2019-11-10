#!/usr/bin/env bash
#SBATCH --job-name=Turygin_Project
#SBATCH --sockets-per-node=1 
#SBATCH --partition=slurm_shortgpu
#SBATCH --ntasks=1 --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-00:00:30

cd $SLURM_SUBMIT_DIR
./jacobian_cpu inputArray.inp output.txt 0.1
./jacobian_cpu inputArray128-8.inp output128-8.txt 0.1
./jacobian_cpu inputArray384-8.inp output384-8.txt 0.1
./jacobian_cpu inputArray512-8.inp output512-8.txt 0.1
./jacobian_cpu inputArray700-5.inp output700-5.txt 0.1
./jacobian_cpu inputArray2048-8.inp output2048-8.txt 0.1

