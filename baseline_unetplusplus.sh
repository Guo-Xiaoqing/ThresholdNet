#!/bin/bash
#SBATCH -J UNetplusplus_384_256_CVC
#SBATCH -o UNetplusplus_384_256_CVC.out               
#SBATCH -e UNetplusplus_384_256_CVC.err
#SBATCH --gres=gpu:1
#SBATCH -w node4

echo "Submitted from:"$SLURM_SUBMIT_DIR" on node:"$SLURM_SUBMIT_HOST
echo "Running on node "$SLURM_JOB_NODELIST 
echo "Allocate Gpu Units:"$CUDA_VISIBLE_DEVICES

source /home/xiaoqiguo2/.bashrc

conda activate torch

cd /home/xiaoqiguo2/Threshold/experiment/baseline_UNetplusplus/
python ./train.py
