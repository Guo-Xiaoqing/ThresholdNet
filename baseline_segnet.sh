#!/bin/bash
#SBATCH -J SegNet_256_256_WCE_cross1
#SBATCH -o SegNet_256_256_WCE_cross1.out               
#SBATCH -e SegNet_256_256_WCE_cross1.err
#SBATCH --gres=gpu:1
#SBATCH -w node4

echo "Submitted from:"$SLURM_SUBMIT_DIR" on node:"$SLURM_SUBMIT_HOST
echo "Running on node "$SLURM_JOB_NODELIST 
echo "Allocate Gpu Units:"$CUDA_VISIBLE_DEVICES

source /home/xiaoqiguo2/.bashrc

conda activate torch

cd /home/xiaoqiguo2/Threshold/experiment/baseline_SegNet/
python ./train.py
