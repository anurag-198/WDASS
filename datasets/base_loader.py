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

Generic dataloader base class
"""
import os
import sys
import glob
import numpy as np
import torch

import PIL.Image
#from PIL import Image
from torch.utils import data
from config import cfg
from datasets import uniform
from runx.logx import logx
from utils.misc import tensor_to_pil
import datasets.edge_utils as edge_utils
from torchvision import transforms
import imageio

import random

num_classes = 19
ignore_label = 255

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153,
           153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60,
           255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask

class BaseLoader(data.Dataset):
    def __init__(self, quality, mode, joint_transform_list, img_transform,
                 label_transform):

        super(BaseLoader, self).__init__()
        self.quality = quality
        self.mode = mode
        self.joint_transform_list = joint_transform_list
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.train = mode == 'train'
        self.id_to_trainid = {}
        self.id_to_trainidsynthia = {1:10, 2:2, 3:0, 4:1, 5:4, 6:8, 7:5, 8:13, 
                            9:7, 10:11, 11:18, 12:17, 15:6, 16:9, 17:12, 
                            18:14, 19:15, 20:16, 21:3}
        self.centroids = None
        self.all_imgs = None
        self.drop_mask = np.zeros((1024, 2048))
        self.drop_mask[15:840, 14:2030] = 1.0
        self.sample_size = None
        self.psl = None
        self.syn_len = 0
        self.weak_label = None
        self.synthia = False

    def build_epoch(self):
        """
        For class uniform sampling ... every epoch, we want to recompute
        which tiles from which images we want to sample from, so that the
        sampling is uniformly random.
        """
        if self.train : 
            self.imgs, self.fine_imgs, self.coarse_imgs = uniform.build_epoch(self.all_imgs,
                                        self.centroids,
                                        self.num_classes,
                                        self.train,
                                        self.sample_size,
                                        self.psl,
                                        self.weak_label,
                                        self.synthia)
            self.syn_len = len(self.fine_imgs)
            self.coarse_len = len(self.coarse_imgs)
        else : 
            self.imgs = uniform.build_epoch(self.all_imgs,
                                        self.centroids,
                                        self.num_classes,
                                        self.train,
                                        self.sample_size,
                                        self.psl, self.weak_label,
                                        self.synthia)

    @staticmethod
    def find_images(img_root, mask_root, img_ext, mask_ext):
        """
        Find image and segmentation mask files and return a list of
        tuples of them.
        """
        img_path = '{}/*.{}'.format(img_root, img_ext)
        imgs = glob.glob(img_path)
        items = []
        for full_img_fn in imgs:
            img_dir, img_fn = os.path.split(full_img_fn)
            img_name, _ = os.path.splitext(img_fn)
            full_mask_fn = '{}.{}'.format(img_name, mask_ext)
            full_mask_fn = os.path.join(mask_root, full_mask_fn)
            assert os.path.exists(full_mask_fn)
            items.append((full_img_fn, full_mask_fn))
        return items

    def disable_coarse(self):
        pass

    def colorize_mask(self, image_array):
        """
        Colorize the segmentation mask
        """
        new_mask = PIL.Image.fromarray(image_array.astype(np.uint8)).convert('P')
        new_mask.putpalette(self.color_mapping)
        return new_mask

    def dump_images(self, img_name, mask, centroid, class_id, img):
        #img = tensor_to_pil(img) 
        outdir = 'new_dump_imgs_{}'.format(self.mode)
        os.makedirs(outdir, exist_ok=True)
        if centroid is not None:
            dump_img_name = '{}_{}'.format(self.trainid_to_name[class_id],
                                           img_name)
        else:
            dump_img_name = img_name
        out_img_fn = os.path.join(outdir, dump_img_name + '.png')
        out_msk_fn = os.path.join(outdir, dump_img_name + '_mask.png')
        out_raw_fn = os.path.join(outdir, dump_img_name + '_mask_raw.png')
        mask_img = self.colorize_mask(np.array(mask))
        img.save(out_img_fn)
        mask_img.save(out_msk_fn)

    def do_transforms(self, img, mask, centroid, img_name, class_id):
        """
        Do transformations to image and mask

        :returns: image, mask
        """
        scale_float = 1.0

        if self.joint_transform_list is not None:
            for idx, xform in enumerate(self.joint_transform_list):
                if idx == 0 and centroid is not None:
                    # HACK! Assume the first transform accepts a centroid
                    outputs = xform(img, mask, centroid)
                else:
                    outputs = xform(img, mask)

                if len(outputs) == 3:
                    img, mask, scale_float = outputs
                else:
                    img, mask = outputs

        if self.img_transform is not None:
            img = self.img_transform(img)

        if cfg.DATASET.DUMP_IMAGES:
            self.dump_images(img_name, mask, centroid, class_id, img)

        if self.label_transform is not None:
            mask = self.label_transform(mask)

        return img, mask, scale_float

    def doDACS(self, img, syn_img, mask, syn_mask) : 
        classes = np.unique(syn_mask)
        n_classes = classes.shape[0]
        classes = np.random.choice(classes, int((n_classes + n_classes%2)/4), replace=False)
        mask_obj = np.full(syn_mask.shape, False)
        for cl in classes : 
            mask_obj = (syn_mask == cl) + mask_obj 
        mask[mask_obj] = syn_mask[mask_obj]
        img_ = np.array(img)
        syn_img_ = np.array(syn_img)
        img_[mask_obj,:] = syn_img_[mask_obj,:]
        img_ = PIL.Image.fromarray(img_)
        return img_, mask
    
    def read_images_synthia(self, img_path, mask_path, dacs, mask_out=False):
        img = PIL.Image.open(img_path).convert('RGB')
        mask = imageio.imread(mask_path, format='PNG-FI')[:,:,0]
        mask = PIL.Image.fromarray(np.uint8(mask))
        
        if "cityscapes" in mask_path : #### assuming we will not use cityscapes fine data for training
            coarse = True
        else :
            coarse = False
        
        img_name = os.path.splitext(os.path.basename(img_path))[0]        
        mask = np.array(mask)
        #print("before ", np.unique(mask), img_path)
        mask = mask.copy()
        for k, v in self.id_to_trainidsynthia.items():  ## this is for both GTA and Cityscaprs
            binary_mask = (mask == k) #+ (gtCoarse == k)
            mask[binary_mask] = v

        mask1 = mask < 0
        mask2 = mask > 18 ### after you do this 
        mask_fin = np.logical_or(mask1, mask2)
        mask[mask_fin] = 255

        #print("after ", np.unique(mask))
        
        mask = PIL.Image.fromarray(mask.astype(np.uint8))
        return img, mask, coarse, img_name


    def read_images(self, img_path, mask_path, dacs, mask_out=False):
        img = PIL.Image.open(img_path).convert('RGB')
        if mask_path is None or mask_path == '':
            w, h = img.size
            mask = np.zeros((h, w))
            logx.msg(" mask is empty, taking default value ####")
            sys.exit()
        else:
            mask = PIL.Image.open(mask_path)
        
        if "cityscapes" in mask_path : #### assuming we will not use cityscapes fine data for training
            coarse = True
        else :
            coarse = False
        
        drop_out_mask = None
        # This code is specific to cityscapes

        img_name = os.path.splitext(os.path.basename(img_path))[0]
        
        if "GTA" in mask_path :
            mask = mask.resize((1280, 720), resample=PIL.Image.NEAREST)
            img = img.resize((1280, 720), resample=PIL.Image.BILINEAR)

        if "cityscapes" in mask_path and self.train:
            #mask = mask.resize((1024, 512), resample=PIL.Image.NEAREST)
            img = img.resize((1024, 512), resample=PIL.Image.BILINEAR)

        flag_wl = False
        if self.weak_label is not None :
            flag_wl = 'coarse' in self.weak_label or 'image' in self.weak_label 
        if self.train and self.weak_label is not None and flag_wl and 'cityscapes' in mask_path : ### point already is in 1024,512
            mask = mask.resize((1024, 512), resample=PIL.Image.NEAREST)
      
        mask = np.array(mask)
        
        if (mask_out):
            mask = self.drop_mask * mask

        mask = mask.copy()
        for k, v in self.id_to_trainid.items():  ## this is for both GTA and Cityscaprs
            binary_mask = (mask == k) #+ (gtCoarse == k)
            mask[binary_mask] = v

        mask1 = mask < 0
        mask2 = mask > 33
        mask_fin = np.logical_or(mask1, mask2)
        mask[mask_fin] = 255
        
        mask = PIL.Image.fromarray(mask.astype(np.uint8))
        return img, mask, coarse, img_name

    ### doing strong augmentation on coarse image by adding samples of synscapes data in it.

    def __getitem__(self, index):  
        """
        Generate data:

        :return:
        - image: image, tensor
        - mask: mask, tensor
        - image_name: basename of file, string
        """
        dacs = False
        # Pick an image, fill in defaults if not using class uniform
        if len(self.imgs[index]) == 2:
            img_path, mask_path = self.imgs[index]
            centroid = None
            class_id = None
        else:
            print("got an emergency")
            print(self.imgs[index])
            img_path, mask_path, centroid, class_id = self.imgs[index]

        if "cityscapes" in img_path :
            dacs = True
        
        mask_out = cfg.DATASET.MASK_OUT_CITYSCAPES and \
            cfg.DATASET.CUSTOM_COARSE_PROB is not None and \
            'refinement' in mask_path

        
        if self.train:
            img_path = self.data_root + img_path
            mask_path = self.data_root + mask_path

        if 'SYNTHIA' not in img_path :
            img, mask, coarse, img_name = self.read_images(img_path, mask_path, dacs,
                                                mask_out=mask_out)
        else :
            img, mask, coarse, img_name = self.read_images_synthia(img_path, mask_path, dacs,
                                                mask_out=mask_out)
        ######################################################################
        # Thresholding is done when using coarse-labelled Cityscapes images
        ######################################################################
        #print("on reading ", img.shape, mask.shape)
        
        if 'refinement' in mask_path:
            
            mask = np.array(mask)
            prob_mask_path = mask_path.replace('.png', '_prob.png')
            # put it in 0 to 1
            prob_map = np.array(PIL.Image.open(prob_mask_path)) / 255.0
            prob_map_threshold = (prob_map < cfg.DATASET.CUSTOM_COARSE_PROB)
            mask[prob_map_threshold] = cfg.DATASET.IGNORE_LABEL
            mask = PIL.Image.fromarray(mask.astype(np.uint8))

        img, mask, scale_float = self.do_transforms(img, mask, centroid,
                                                    img_name, class_id)

        
        _edgemap = mask.numpy()
        _edgemap = edge_utils.mask_to_onehot(_edgemap, num_classes)

        _edgemap = edge_utils.onehot_to_binary_edges(_edgemap, 2, num_classes)
        edgemap = torch.from_numpy(_edgemap).float()

        dump_images = False
        if dump_images:
            self.dump_images(img_name, mask, None, None, img)
            #outdir = 'dump_imgs_{}'.format(self.mode)
            #os.makedirs(outdir, exist_ok=True)  
            #out_img_fn = os.path.join(outdir, img_name + '.png')
            #out_msk_fn = os.path.join(outdir, img_name + '_mask.png')
            #edge_msk_fn = os.path.join(outdir, img_name + '_edge.png')
            #mask_img = colorize_mask(np.array(mask))
            #edge_msk = transforms.ToPILImage()(edgemap.squeeze())
            #mask_img.save(out_msk_fn)

        return img, mask, edgemap, coarse, img_name, scale_float

    def __len__(self):
        return len(self.imgs)

    def calculate_weights(self):
        raise BaseException("not supported yet")
