#!/bin/bash
#SBATCH -J CVC_ThresholdNet_CGMMIx
#SBATCH -o CVC_ThresholdNet_CGMMIx.out               
#SBATCH -e error.err
#SBATCH --gres=gpu:2
#SBATCH -w node4

echo "Submitted from:"$SLURM_SUBMIT_DIR" on node:"$SLURM_SUBMIT_HOST
echo "Running on node "$SLURM_JOB_NODELIST 
echo "Allocate Gpu Units:"$CUDA_VISIBLE_DEVICES

source /home/xiaoqiguo2/.bashrc

conda activate torch

cd ~/ThresholdNet/experiment/Ours_ThresholdNet_CGMMIx/
python ./train.py
