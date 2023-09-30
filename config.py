"""
# Code adapted from:
# https://github.com/facebookresearch/Detectron/blob/master/detectron/core/config.py

Source License
# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
#
# Based on:
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
"""
##############################################################################
# Config
##############################################################################


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import re
import torch

from utils.attr_dict import AttrDict
from runx.logx import logx


__C = AttrDict()
cfg = __C
__C.GLOBAL_RANK = 0
__C.EPOCH = 0
# Absolute path to a location to keep some large files, not in this dir.
__C.ASSETS_PATH = '/BS/ZeroLabelSemanticSegmentation/work/data'

# Use class weighted loss per batch to increase loss for low pixel count classes per batch
__C.BATCH_WEIGHTING = False

# Border Relaxation Count
__C.BORDER_WINDOW = 1
# Number of epoch to use before turn off border restriction
__C.REDUCE_BORDER_EPOCH = -1
# Comma Seperated List of class id to relax
__C.STRICTBORDERCLASS = None
# Where output results get written
__C.RESULT_DIR = None

__C.OPTIONS = AttrDict()
__C.OPTIONS.TEST_MODE = False
__C.OPTIONS.INIT_DECODER = False
__C.OPTIONS.TORCH_VERSION = None

__C.TRAIN = AttrDict()
__C.TRAIN.RANDOM_BRIGHTNESS_SHIFT_VALUE = 10
__C.TRAIN.FP16 = False

#Attribute Dictionary for Dataset
__C.DATASET = AttrDict()
#Cityscapes Dir Location
__C.DATASET.CITYSCAPES_DIR = \
  os.path.join(__C.ASSETS_PATH, 'cityscapes')
__C.DATASET.CITYSCAPES_CUSTOMCOARSE = \
  os.path.join(__C.ASSETS_PATH, 'cityscapes/autolabelled')
__C.DATASET.CENTROID_ROOT = \
  os.path.join(__C.ASSETS_PATH, 'uniform_centroids')
#SDC Augmented Cityscapes Dir Location
__C.DATASET.CITYSCAPES_AUG_DIR = ''
#Mapillary Dataset Dir Location
__C.DATASET.MAPILLARY_DIR = os.path.join(__C.ASSETS_PATH, 'Mapillary/data')
#Kitti Dataset Dir Location
__C.DATASET.KITTI_DIR = ''
#SDC Augmented Kitti Dataset Dir Location
__C.DATASET.KITTI_AUG_DIR = ''
#Camvid Dataset Dir Location
__C.DATASET.CAMVID_DIR = ''
#Number of splits to support
__C.DATASET.CITYSCAPES_SPLITS = 3
__C.DATASET.MEAN = [0.485, 0.456, 0.406]
__C.DATASET.STD = [0.229, 0.224, 0.225]
__C.DATASET.NAME = ''
__C.DATASET.NUM_CLASSES = 0
__C.DATASET.IGNORE_LABEL = 255
__C.DATASET.DUMP_IMAGES = False
__C.DATASET.CLASS_UNIFORM_PCT = 0.5
__C.DATASET.CLASS_UNIFORM_TILE = 1024
__C.DATASET.COARSE_BOOST_CLASSES = None
__C.DATASET.CV = 0
__C.DATASET.COLORIZE_MASK_FN = None
__C.DATASET.CUSTOM_COARSE_PROB = None
__C.DATASET.MASK_OUT_CITYSCAPES = False

# This enables there to always be translation augmentation during random crop
# process, even if image is smaller than crop size.
__C.DATASET.TRANSLATE_AUG_FIX = False
__C.DATASET.LANCZOS_SCALES = False
# Use a center crop of size args.pre_size for mapillary validation
# Need to use this if you want to dump images
__C.DATASET.MAPILLARY_CROP_VAL = False
__C.DATASET.CROP_SIZE = '896'

__C.MODEL = AttrDict()
__C.MODEL.BN = 'regularnorm'
__C.MODEL.BNFUNC = None
__C.MODEL.MSCALE = False
__C.MODEL.THREE_SCALE = False
__C.MODEL.ALT_TWO_SCALE = False
__C.MODEL.EXTRA_SCALES = '0.5,1.5'
__C.MODEL.N_SCALES = None
__C.MODEL.ALIGN_CORNERS = False
__C.MODEL.MSCALE_LO_SCALE = 0.5
__C.MODEL.OCR_ASPP = False
__C.MODEL.SEGATTN_BOT_CH = 256
__C.MODEL.ASPP_BOT_CH = 256
__C.MODEL.MSCALE_CAT_SCALE_FLT = False
__C.MODEL.MSCALE_INNER_3x3 = True
__C.MODEL.MSCALE_DROPOUT = False
__C.MODEL.MSCALE_OLDARCH = False
__C.MODEL.MSCALE_INIT = 0.5
__C.MODEL.ATTNSCALE_BN_HEAD = False
__C.MODEL.GRAD_CKPT = False

WEIGHTS_PATH = os.path.join(__C.ASSETS_PATH, 'seg_weights')
__C.MODEL.WRN38_CHECKPOINT = \
    os.path.join(WEIGHTS_PATH, 'wider_resnet38.pth.tar')
__C.MODEL.WRN41_CHECKPOINT = \
    os.path.join(WEIGHTS_PATH, 'wider_resnet41_cornflower_sunfish.pth')
__C.MODEL.X71_CHECKPOINT = \
    os.path.join(WEIGHTS_PATH, 'aligned_xception71.pth')
__C.MODEL.HRNET_CHECKPOINT = \
    os.path.join(WEIGHTS_PATH, 'hrnetv2_w48_imagenet_pretrained.pth')

__C.LOSS = AttrDict()
# Weight for OCR aux loss
__C.LOSS.OCR_ALPHA = 0.4
# Use RMI for the OCR aux loss
__C.LOSS.OCR_AUX_RMI = False
# Supervise the multi-scale predictions directly
__C.LOSS.SUPERVISED_MSCALE_WT = 0

__C.MODEL.OCR = AttrDict()
__C.MODEL.OCR.MID_CHANNELS = 512
__C.MODEL.OCR.KEY_CHANNELS = 256
__C.MODEL.OCR_EXTRA = AttrDict()
__C.MODEL.OCR_EXTRA.FINAL_CONV_KERNEL = 1
__C.MODEL.OCR_EXTRA.STAGE1 = AttrDict()
__C.MODEL.OCR_EXTRA.STAGE1.NUM_MODULES = 1
__C.MODEL.OCR_EXTRA.STAGE1.NUM_RANCHES = 1
__C.MODEL.OCR_EXTRA.STAGE1.BLOCK = 'BOTTLENECK'
__C.MODEL.OCR_EXTRA.STAGE1.NUM_BLOCKS = [4]
__C.MODEL.OCR_EXTRA.STAGE1.NUM_CHANNELS = [64]
__C.MODEL.OCR_EXTRA.STAGE1.FUSE_METHOD = 'SUM'
__C.MODEL.OCR_EXTRA.STAGE2 = AttrDict()
__C.MODEL.OCR_EXTRA.STAGE2.NUM_MODULES = 1
__C.MODEL.OCR_EXTRA.STAGE2.NUM_BRANCHES = 2
__C.MODEL.OCR_EXTRA.STAGE2.BLOCK = 'BASIC'
__C.MODEL.OCR_EXTRA.STAGE2.NUM_BLOCKS = [4, 4]
__C.MODEL.OCR_EXTRA.STAGE2.NUM_CHANNELS = [48, 96]
__C.MODEL.OCR_EXTRA.STAGE2.FUSE_METHOD = 'SUM'
__C.MODEL.OCR_EXTRA.STAGE3 = AttrDict()
__C.MODEL.OCR_EXTRA.STAGE3.NUM_MODULES = 4
__C.MODEL.OCR_EXTRA.STAGE3.NUM_BRANCHES = 3
__C.MODEL.OCR_EXTRA.STAGE3.BLOCK = 'BASIC'
__C.MODEL.OCR_EXTRA.STAGE3.NUM_BLOCKS = [4, 4, 4]
__C.MODEL.OCR_EXTRA.STAGE3.NUM_CHANNELS = [48, 96, 192]
__C.MODEL.OCR_EXTRA.STAGE3.FUSE_METHOD = 'SUM'
__C.MODEL.OCR_EXTRA.STAGE4 = AttrDict()
__C.MODEL.OCR_EXTRA.STAGE4.NUM_MODULES = 3
__C.MODEL.OCR_EXTRA.STAGE4.NUM_BRANCHES = 4
__C.MODEL.OCR_EXTRA.STAGE4.BLOCK = 'BASIC'
__C.MODEL.OCR_EXTRA.STAGE4.NUM_BLOCKS = [4, 4, 4, 4]
__C.MODEL.OCR_EXTRA.STAGE4.NUM_CHANNELS = [48, 96, 192, 384]
__C.MODEL.OCR_EXTRA.STAGE4.FUSE_METHOD = 'SUM'


def torch_version_float():
    version_str = torch.__version__
    version_re = re.search(r'^([0-9]+\.[0-9]+)', version_str)
    if version_re:
        version = float(version_re.group(1))
        logx.msg(f'Torch version: {version}, {version_str}')
    else:
        version = 1.0
        logx.msg(f'Can\'t parse torch version ({version}), assuming {version}')
    return version


def assert_and_infer_cfg(args, make_immutable=True, train_mode=True):
    """Call this function in your script after you have finished setting all cfg
    values that are necessary (e.g., merging a config from a file, merging
    command line config options, etc.). By default, this function will also
    mark the global cfg as immutable to prevent changing the global cfg
    settings during script execution (which can lead to hard to debug errors
    or code that's harder to understand than is necessary).
    """

    __C.OPTIONS.TORCH_VERSION = torch_version_float()

    if hasattr(args, 'syncbn') and args.syncbn:
        if args.apex:
            import apex
            __C.MODEL.BN = 'apex-syncnorm'
            __C.MODEL.BNFUNC = apex.parallel.SyncBatchNorm
        else:
            raise Exception('No Support for SyncBN without Apex')
    else:
        __C.MODEL.BNFUNC = torch.nn.BatchNorm2d
        print('Using regular batch norm')

    if not train_mode:
        cfg.immutable(True)
        return

    

    cfg.DATASET.NAME = args.dataset
    

   

   
    cfg.DATASET.CV = args.cv
    # Total number of splits
    cfg.DATASET.CV_SPLITS = 3

  

   

   
    def str2list(s):
        alist = s.split(',')
        alist = [float(x) for x in alist]
        return alist

    if args.n_scales:
        cfg.MODEL.N_SCALES = str2list(args.n_scales)
        logx.msg('n scales {}'.format(cfg.MODEL.N_SCALES))


    cfg.RESULT_DIR = args.result_dir


    __C.DATASET.CROP_SIZE = args.crop_size

    __C.GLOBAL_RANK = args.global_rank

    if make_immutable:
        cfg.immutable(True)


def update_epoch(epoch):
    # Update EPOCH CTR
    cfg.immutable(False)
    cfg.EPOCH = epoch
    cfg.immutable(True)


def update_dataset_cfg(num_classes, ignore_label):
    cfg.immutable(False)
    cfg.DATASET.NUM_CLASSES = num_classes
    cfg.DATASET.IGNORE_LABEL = ignore_label
    logx.msg('num_classes = {}'.format(num_classes))
    cfg.immutable(True)


def update_dataset_inst(dataset_inst):
    cfg.immutable(False)
    cfg.DATASET_INST = dataset_inst
    cfg.immutable(True)
