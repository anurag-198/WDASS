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

import numpy as np

from runx.logx import logx

pbar = None

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
        pth = "files/cityscapes.npy"
        co = np.load(pth)
        co = co.tolist()
        if 'point' in weak_label :
            print("using point labels ")
            co = modifypt(co)
        elif 'label' in weak_label :
            raise NotImplementedError("label not implemented")
        else :
            print("using coarse labels ---")
        print("loaded " + pth + str(sample_size) + ".npy with length", len(co))
        co_orig = co
    fi = []
    if fine is not None :
        if synthia :
            pth1 = "files/synthia.npy"
        else :
            pth1 = "files/gta5.npy"
        fi = np.load(pth1)
        fi = fi.tolist()
        fi = change(fi)
        print("loaded " + pth1 + str(sample_size) + ".npy with length", len(fi))
        co = fi
    
    return co, fi, co_orig

def build_epoch(imgs, centroids, num_classes, train, sample_size, psl, weak_label, synthia):

    if not train: 
        return imgs

    imgs_uniform, fine_sample, coarse_sample = get_imgs(sample_size, weak_label, synthia)

    logx.msg('Total images (final) : {}'.format(str(len(imgs_uniform))))

    if train :
        return imgs_uniform, fine_sample, coarse_sample
    else : 
        return imgs_uniform
