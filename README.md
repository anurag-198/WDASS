# Weakly-Supervised Domain Adaptive Semantic Segmentation with Prototypical Contrastive Learning
[![Framework](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&logo=PyTorch&logoColor=white)](https://pytorch.org/)


This is the official repository accompanying the CVPR paper:

[Anurag Das](https://anurag-198.github.io/), [Yongqin Xian](https://xianyongqin.github.io/), [Dengxin Dai](https://vas.mpi-inf.mpg.de/dengxin/), and [Bernt Schiele](https://scholar.google.com/citations?user=z76PBfYAAAAJ&hl=en). **Weakly-Supervised Domain Adaptive Semantic Segmentation with Prototypical Contrastive Learning**. CVPR 2023.

[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Das_Weakly-Supervised_Domain_Adaptive_Semantic_Segmentation_With_Prototypical_Contrastive_Learning_CVPR_2023_paper.pdf) | [Video](https://www.youtube.com/watch?v=Arg8p0Zrf9A) | [Supplemental](https://openaccess.thecvf.com/content/CVPR2023/supplemental/Das_Weakly-Supervised_Domain_Adaptive_CVPR_2023_supplemental.pdf)


## Usage

#### For Conda:
Create a conda environment using the provided environment.yml file:

```bash
conda env create -f environment.yml
conda activate cu11
```
#### For Pip:
Create a virtual environment using the provided requirements.txt file:

```bash
python3 -m venv cu11
source cu11/bin/activate
pip install -r requirements.txt
```

### Experiment setup:

#### Preparing the data:
1. The dataset supported are Cityscapes, GTA5 and Synthia. Download the Cityscapes dataset from [here](https://www.cityscapes-dataset.com/). The dataset directory should have the following structure:

```
data
 SYNTHIA/
 GTA5/
 cityscapes/
 ├── leftImg8bit_trainextra/
 │   └── leftImg8bit/
 │    └── train/
 │       ├── [subfolder_1]/
 │       │   └── [image_files]
 │       ├── [subfolder_2]/
 │       │   └── [image_files]
 │       └── ...
 ├── gtFine_trainextra/
 │   └── gtFine/
 │    └── train/
 │       ├── [subfolder_1]/
 │       │   └── [mask_files]
 │       ├── [subfolder_2]/
 │       │   └── [mask_files]
 │       └── ...
 ├── leftImg8bit_trainvaltest/
 │   └── leftImg8bit/
 │    └── train/
 │    └── val/
 │    └── train_extra/
 ├── gtFine_trainvaltest/
 │   └── gtFine/
 │    └── train/
 │    └── val/
 │    └── train_extra/
 ├── gtCoarse/
 │    └── train/
 │    └── val/
 │    └── train_extra/
```

2. change the ```__C.ASSETS_PATH``` in ```config.py``` to the path of the dataset directory.

3. **(Optional)** For using the point annotation ```--weak_label point``` during training, first generate the ground truth point annotations using the following command:

```bash
python datasets/utils.py
```

#### Training:
1. Train segmentation model on synthetic data (GTA5) to obtain GTA pretrained weights
2. Use GTA5 pretrained model to start training for adaptation using below -  
```bash
python train.py --dataset cityscapes --result_dir logs/ --multiprocessing_distributed  --use_contrast  --bn_buffer --weak_label coarse --use_wl --imloss --improto --resume "pretrained gta weights"
python train.py --dataset cityscapes --result_dir logs/ --multiprocessing_distributed  --use_contrast  --bn_buffer --weak_label point --use_wl --imloss --improto --resume "pretrained gta weights"
python train.py --dataset cityscapes --result_dir logs/ --multiprocessing_distributed  --use_contrast  --bn_buffer --weak_label image --imloss --improto --resume "pretrained gta weights"
```

#### Validation:
You can use the following command:

```bash
python train.py --dataset cityscapes --eval val --n_scales 0.5,1,2 --arch deepv2.DeepV2R101 --result_dir  logs/coarse --multiprocessing_distributed --snapshot 'location to checkpoint'
```

#### Trained weights for validation 
You can download trained weights for coarse, point and image labels from [here](https://github.com/anurag-198/WDASS/releases/tag/v1.0.0)

## Citation

If you find our work useful, please consider citing our paper:

```
@InProceedings{Das_2023_CVPR,
    author    = {Das, Anurag and Xian, Yongqin and Dai, Dengxin and Schiele, Bernt},
    title     = {Weakly-Supervised Domain Adaptive Semantic Segmentation With Prototypical Contrastive Learning},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {15434-15443}
}
```
