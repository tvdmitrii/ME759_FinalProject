#!/usr/bin/env bash
#SBATCH --job-name=Turygin_Project
#SBATCH --sockets-per-node=1 
#SBATCH --partition=slurm_shortgpu
#SBATCH --ntasks=1 --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-00:01:00

cd $SLURM_SUBMIT_DIR
./jacobian inputArray.inp output.txt 0.1
./jacobian inputArray128-8.inp output128-8.txt 0.1
./jacobian inputArray384-8.inp output384-8.txt 0.1
./jacobian inputArray512-8.inp output512-8.txt 0.1
./jacobian inputArray700-5.inp output700-5.txt 0.1
./jacobian inputArray2048-8.inp output2048-8.txt 0.1
./jacobian inputArray4096-8.inp output4096-8.txt 0.1
./jacobian inputArray8192-8.inp output8192-8.txt 0.1
./jacobian inputArray16384-8.inp output16384-8.txt 0.1
./jacobian inputArray32768-8.inp output32768-8.txt 0.1
./jacobian inputArray65536-8.inp output65536-8.txt 0.1
./jacobian inputArray131072-8.inp output131072-8.txt 0.1
./jacobian inputArray262144-8.inp output262144-8.txt 0.1
./jacobian inputArray524288-8.inp output524288-8.txt 0.1
./jacobian inputArray1048576-8.inp output1048576-8.txt 0.1

