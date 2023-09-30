"""
Copyright 2020 Nvidia Corporation

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.


Uniform sampling of classes.
For all images, for all classes, generate centroids around which to sample.

All images are divided into tiles.
For each tile, a class can be present or not. If it is
present, calculate the centroid of the class and record it.

We would like to thank Peter Kontschieder for the inspiration of this idea.
"""

import sys
import os
import json
import numpy as np

import torch

from collections import defaultdict
from scipy.ndimage.measurements import center_of_mass
from PIL import Image
from tqdm import tqdm
from config import cfg
from runx.logx import logx

from os import listdir
from os.path import isfile, join

pbar = None

class Point():
    """
    Point Class For X and Y Location
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y


def calc_tile_locations(tile_size, image_size):
    """
    Divide an image into tiles to help us cover classes that are spread out.
    tile_size: size of tile to distribute
    image_size: original image size
    return: locations of the tiles
    """
    image_size_y, image_size_x = image_size
    locations = []
    for y in range(image_size_y // tile_size):
        for x in range(image_size_x // tile_size):
            x_offs = x * tile_size
            y_offs = y * tile_size
            locations.append((x_offs, y_offs))
    return locations

def class_centroids_image(item, tile_size, num_classes, id2trainid):
    """
    For one image, calculate centroids for all classes present in image.
    item: image, image_name
    tile_size:
    num_classes:
    id2trainid: mapping from original id to training ids
    return: Centroids are calculated for each tile.
    """
    image_fn, label_fn = item
    centroids = defaultdict(list)
    mask = np.array(Image.open(label_fn))
    image_size = mask.shape
    tile_locations = calc_tile_locations(tile_size, image_size)

    drop_mask = np.zeros((1024,2048))
    drop_mask[15:840, 14:2030] = 1.0

    #####
    if(cfg.DATASET.CITYSCAPES_CUSTOMCOARSE in label_fn):
            gtCoarse_mask_path = label_fn.replace(cfg.DATASET.CITYSCAPES_CUSTOMCOARSE, os.path.join(cfg.DATASET.CITYSCAPES_DIR, 'gtCoarse/gtCoarse') )
            gtCoarse_mask_path = gtCoarse_mask_path.replace('leftImg8bit','gtCoarse_labelIds')          
            gtCoarse=np.array(Image.open(gtCoarse_mask_path))
    ####

    mask_copy = mask.copy()
    if id2trainid:
        for k, v in id2trainid.items():
            binary_mask = (mask_copy == k)
            #This should only apply to auto labelled images
            if ('refinement' in label_fn) and cfg.DROPOUT_COARSE_BOOST_CLASSES != None and v in cfg.DROPOUT_COARSE_BOOST_CLASSES and binary_mask.sum() > 0:
                binary_mask += (gtCoarse == k)
                binary_mask[binary_mask >= 1] = 1
                mask[binary_mask] = gtCoarse[binary_mask]
            mask[binary_mask] = v

    for x_offs, y_offs in tile_locations:
        patch = mask[y_offs:y_offs + tile_size, x_offs:x_offs + tile_size]
        for class_id in range(num_classes):
            if class_id in patch:
                patch_class = (patch == class_id).astype(int)
                centroid_y, centroid_x = center_of_mass(patch_class)
                centroid_y = int(centroid_y) + y_offs
                centroid_x = int(centroid_x) + x_offs
                centroid = (centroid_x, centroid_y)
                centroids[class_id].append((image_fn, label_fn, centroid,
                                            class_id))
    pbar.update(1)
    return centroids


def pooled_class_centroids_all(items, num_classes, id2trainid, tile_size=1024):
    """
    Calculate class centroids for all classes for all images for all tiles.
    items: list of (image_fn, label_fn)
    tile size: size of tile
    returns: dict that contains a list of centroids for each class
    """
    from multiprocessing.dummy import Pool
    from functools import partial
    pool = Pool(80)
    global pbar
    pbar = tqdm(total=len(items), desc='pooled centroid extraction', file=sys.stdout)
    class_centroids_item = partial(class_centroids_image,
                                   num_classes=num_classes,
                                   id2trainid=id2trainid,
                                   tile_size=tile_size)

    centroids = defaultdict(list)
    new_centroids = pool.map(class_centroids_item, items)
    pool.close()
    pool.join()

    # combine each image's items into a single global dict
    for image_items in new_centroids:
        for class_id in image_items:
            centroids[class_id].extend(image_items[class_id])
    return centroids


def unpooled_class_centroids_all(items, num_classes, id2trainid,
                                 tile_size=1024):
    """
    Calculate class centroids for all classes for all images for all tiles.
    items: list of (image_fn, label_fn)
    tile size: size of tile
    returns: dict that contains a list of centroids for each class
    """
    centroids = defaultdict(list)
    global pbar
    pbar = tqdm(total=len(items), desc='centroid extraction', file=sys.stdout)
    for image, label in items:
        new_centroids = class_centroids_image(item=(image, label),
                                              tile_size=tile_size,
                                              num_classes=num_classes,
                                              id2trainid=id2trainid)
        for class_id in new_centroids:
            centroids[class_id].extend(new_centroids[class_id])
    return centroids

def class_centroids_all(items, num_classes, id2trainid, tile_size=1024):
    """
    intermediate function to call pooled_class_centroid
    """
    pooled_centroids = pooled_class_centroids_all(items, num_classes,
                                                  id2trainid, tile_size)
    # pooled_centroids = unpooled_class_centroids_all(items, num_classes,
    #                                                id2trainid, tile_size)
    return pooled_centroids


def random_sampling(alist, num):
    """
    Randomly sample num items from the list
    alist: list of centroids to sample from
    num: can be larger thansudo du -a /dir/ | sort -n -r | head -n 20 the list and if so, then wrap around
    return: class uniform samples from the list
    """
    sampling = []
    len_list = len(alist)
    assert len_list, 'len_list is zero!'
    indices = np.arange(len_list)
    np.random.shuffle(indices)

    for i in range(num):
        item = alist[indices[i % len_list]]
        sampling.append(item)
    return sampling


def build_centroids(imgs, num_classes, train, cv=None, coarse=False,
                    custom_coarse=False, id2trainid=None):
    """
    The first step of uniform sampling is to decide sampling centers.
    The idea is to divide each image into tiles and within each tile,
    we compute a centroid for each class to indicate roughly where to
    sample a crop during training.

    This function computes these centroids and returns a list of them.
    """
    if not (cfg.DATASET.CLASS_UNIFORM_PCT and train):
        return []

    centroid_fn = cfg.DATASET.NAME
    
    if coarse or custom_coarse:
        if coarse:
            centroid_fn += '_coarse'
        if custom_coarse:
            centroid_fn += '_customcoarse_final'
    else:
        centroid_fn += '_cv{}'.format(cv)
    centroid_fn += '_tile{}.json'.format(cfg.DATASET.CLASS_UNIFORM_TILE)
    json_fn = os.path.join(cfg.DATASET.CENTROID_ROOT,
                           centroid_fn)
    if os.path.isfile(json_fn):
        logx.msg('Loading centroid file {}'.format(json_fn))
        with open(json_fn, 'r') as json_data:
            centroids = json.load(json_data)
        centroids = {int(idx): centroids[idx] for idx in centroids}
        logx.msg('Found {} centroids'.format(len(centroids)))
    else:
        logx.msg('Didn\'t find {}, so building it'.format(json_fn))

        if cfg.GLOBAL_RANK==0:

            os.makedirs(cfg.DATASET.CENTROID_ROOT, exist_ok=True)
            # centroids is a dict (indexed by class) of lists of centroids
            centroids = class_centroids_all(
                imgs,
                num_classes,
                id2trainid=id2trainid)
            with open(json_fn, 'w') as outfile:
                json.dump(centroids, outfile, indent=4)

        # wait for everyone to be at the same point
        torch.distributed.barrier()

        #  GPUs (except rank0) read in the just-created centroid file
        if cfg.GLOBAL_RANK != 0:
            msg = f'Expected to find {json_fn}'
            assert os.path.isfile(json_fn), msg
            with open(json_fn, 'r') as json_data:
                centroids = json.load(json_data)
            centroids = {int(idx): centroids[idx] for idx in centroids}
        
    return centroids

'''
def get_imgs() : 
    sv = "/BS/ZeroLabelSemanticSegmentation/work/project/semantic-segmentation/logs/train_cityscapes_deepv3x71/coarseRed/code/logs/train_cityscapes_deepv3x71/FT-1000-3/"
    mypath = sv + "imgs" 
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    a=[]
    
    for i in range(len(onlyfiles)) :
        a1 = onlyfiles[i]
        b1 = a1
        b1 = b1[:-15] + "gtCoarse_labelIds.png"
        a1 = sv + "imgs/" +  a1
        b1 = sv + "nt/" + b1 
        c = (a1,b1)
        a.append(c)
    return a 
'''
####### most important segment ##############  /BS/anurag/work/project/semantic-segmentation/logs/train_cityscapes_deepv3x71/shape_syn_psl/code/datasets/uniform.py
def extend(co, fi) :
    lenco = len(co)
    lenfi = len(fi)
    times = int(lenfi/lenco) ## making fine more available for shape loss (1 fine, 1 coarse, 1 fine-coarse 1 coarse fine)
    co = co * times
    co.extend(fi)
    return co

def modifypt(imgs) : 
    imgs_ = imgs.copy()
    for i in range(len(imgs_)) : 
        if ("cityscapes" in imgs_[i][1]) :
            sp = imgs_[i][1]
            sp = sp.replace("gtCoarse", "gtPoint2")
            imgs_[i][1] = sp
    return imgs_

def change(imgs) : 
    imgs_ = imgs.copy()
    for i in range(len(imgs_)) : 
        if ("GP030402_frame_" in imgs_[i][1]) :
            sp = imgs_[i][0]
            sp = sp.replace("GP030402_frame_", "")
            sp = sp.replace("_rgb_anon", "")
            imgs_[i][0] = sp
            sp = imgs_[i][1]
            sp = sp.replace("GP030402_frame_", "")
            sp = sp.replace("_gt_labelTrainIds", "")
            imgs_[i][1] = sp
    return imgs_

def get_imgs(sample_size, weak_label, synthia) :
    coarse, fine =  sample_size 
    co = []
    co_orig = []
    if coarse is not None :
        pth = "files/coarse/new/p1/file_2975.npy"
        co = np.load(pth)
        co = co.tolist()
        if 'point' in weak_label :
            print("using point labels ")
            co = modifypt(co)
        elif 'label' in weak_label :
            print("not implemented error---------------")
        else :
            print("using coarse labels ---")
        print("loaded " + pth + str(sample_size) + ".npy with length", len(co))
        co_orig = co
    fi = []
    if fine is not None :
        if synthia :
            pth1 = "files/synthia/new/9400.npy"
        else :
            pth1 = "files/gta/new/25000.npy"
        fi = np.load(pth1)
        fi = fi.tolist()
        fi = change(fi)
        print("loaded " + pth1 + str(sample_size) + ".npy with length", len(fi))
        co = fi
    
    return co, fi, co_orig

def modify(imgs, psl, sample_size) : 
    coarse, fine = sample_size
    imgs_ = imgs.copy()
    for i in range(len(imgs_)) : 
        if ("cityscapes" in imgs_[i][1]) :
            sp = imgs_[i][1].split("/")
            name = sp[-3:]
            fn = "/BS/anurag/work/data/cityscapes/gtFine_trainextra/psl/psl_" +  str(coarse) +  "c_" + str(fine) + "f_shape_" + str(psl) +  "/" + name[0] + "/" + name[1] + "/" + name[2]
            imgs_[i][1] = fn
    return imgs_

def build_epoch(imgs, centroids, num_classes, train, sample_size, psl, weak_label, synthia):
    """
    Generate an epoch of crops using uniform sampling.
    Needs to be called every epoch. 
    Will not apply uniform sampling if not train or class uniform is off.

    Inputs: /BS/anurag/work/project/semantic-segmentation/logs/train_cityscapes_deepv3x71/shape_syn/code/datasets/uniform.py
      imgs - list of imgs
      centroids - list of class centroids 
      num_classes - number of classes
      class_uniform_pct: % of uniform images in one epoch
    Outputs:
      imgs - list of images to use this epoch
    """
    class_uniform_pct = cfg.DATASET.CLASS_UNIFORM_PCT
    if not (train and class_uniform_pct): ## check for val
        return imgs

    logx.msg("Class Uniform Percentage: {}".format(str(class_uniform_pct))) 
    num_epoch = int(len(imgs))

    logx.msg('Class Uniform items per Epoch: {}'.format(str(num_epoch)))
    num_per_class = int((num_epoch * class_uniform_pct) / num_classes)
    class_uniform_count = num_per_class * num_classes
    #num_rand = num_epoch - class_uniform_count
    if sample_size is not None :
        num_rand = sample_size
    else :
        num_rand = num_epoch
    # create random crops
    
    #imgs_uniform = random_sampling(imgs, num_rand)
    imgs_uniform, fine_sample, coarse_sample = get_imgs(sample_size, weak_label, synthia)
    
    print("after ............................")
    if psl is not None:
        logx.msg("modifying the target path " +  str(psl))
        imgs_uniform = modify(imgs_uniform, psl, sample_size)

    logx.msg('Total images (final) : {}'.format(str(len(imgs_uniform))))
    
    ##saving files used ###
    #np.save('logs/train_cityscapes_deepv3x71/FT-1000-3/files.npy', imgs_uniform, allow_pickle=True)
    if train :
        return imgs_uniform, fine_sample, coarse_sample
    else : 
        return imgs_uniform
