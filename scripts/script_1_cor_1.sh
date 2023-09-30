#!/bin/bash
#SBATCH -p gpu22

#SBATCH -o //WeakDomainAdaptation/logs/run-%j.out
#SBATCH -t 3:55:00
#SBATCH --gres gpu:2
#SBATCH -a 1-3%1

cd /BS/SAM/work/WeakDomainAdaptation;

# Make conda available:
eval "$(conda shell.bash hook)"
# Activate a conda environment:
source activate cu11
#python train.py --dataset cityscapes --eval val --n_scales 0.5,1,2 --cv 0 --bs_val 8 --arch deepv2.DeepV2R101 --result_dir  logs/test/test_1 --coarse_sample 2975 --fine_sample 25000 --multiprocessing_distributed --snapshot /BS/WDA/work/data/pretrained/gta.pth

python train.py --dataset cityscapes --result_dir logs/ --multiprocessing_distributed  --use_contrast  --bn_buffer --weak_label coarse --resume /data/pretrained/gta.pth --use_wl --imloss --improto