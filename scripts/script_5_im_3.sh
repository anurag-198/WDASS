#!/bin/bash
#SBATCH -p gpu22

#SBATCH -o /BS/WDA/work/project/weakDomainAdaptation/logs/run-%j.out
#SBATCH -t 3:55:00
#SBATCH --gres gpu:2
#SBATCH -a 1-36%1

cd /BS/VisLang/work/WeakDomainAdaptation;

# Make conda available:
eval "$(conda shell.bash hook)"
# Activate a conda environment:
conda activate /BS/ZeroLabelSemanticSegmentation/work/anaconda3/envs/cu11

python train.py --dataset cityscapes --crop_size 512,512 --bs_trn 4 --poly_exp 0.9 --lr 2.5e-3 --weight_decay 0.0001 --weight_decay 0.0001 --max_iter 150000 --cv 0 --arch deepv2.DeepV2R101 --result_dir logs/final/image_ours_3 --coarse_sample 2975 --fine_sample 25000 --multiprocessing_distributed --edge_wt 10 --use_contrast --contrast_wt 1.0  --bn_buffer --weak_label image --resume /BS/WDA/work/data/pretrained/gta.pth --imloss --improto