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
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, ORF
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
POSSIBILITY OF SUCH DAMAGE.
"""
from __future__ import absolute_import
from __future__ import division

import argparse
import os
from pyexpat import features
import sys
import time
import torch
import zipfile

from PIL import Image
from loss.contrast_loss_intra import ContrastCELoss_intra

#from apex import amp
from runx.logx import logx
from config import assert_and_infer_cfg, update_epoch, cfg
from utils.misc import AverageMeter, prep_experiment, eval_metrics
from utils.misc import ImageDumper
from utils.trnval_utils import eval_minibatch, validate_topn, eval_minibatch_psl, eval_minibatch_slide
from loss.utils import get_loss
from loss.optimizer import get_optimizer, restore_opt, restore_net


from utils.init_func import group_weight_2, group_weight
from utils.lr_policy import WarmUpPolyLR, PolyLR
from utils.prototypes import prototypes

from transforms.dacs_transforms import *

import torch.multiprocessing as mp
import torch.distributed as dist

import random 

import datasets
import network
import numpy as np

import sys
import json

import cv2

from timm.models.layers import DropPath
from torch.nn.modules.dropout import _DropoutNd

from pathlib import Path
from loss.contrast_loss import ContrastCELoss
from loss.image_label import ImageLevelLoss

# Import autoresume module
sys.path.append(os.environ.get('SUBMIT_SCRIPTS', '.'))
AutoResume = None
try:
    from userlib.auto_resume import AutoResume
except ImportError:
    print(AutoResume)


# Argument Parser
parser = argparse.ArgumentParser(description='Semantic Segmentation')
parser.add_argument('--lr', type=float, default=0.00025)
parser.add_argument('--arch', type=str, default='deepv3.DeepWV3Plus',
                    help='Network architecture. We have DeepSRNX50V3PlusD (backbone: ResNeXt50) \
                    and deepWV3Plus (backbone: WideResNet38).')
parser.add_argument('--dataset', type=str, default='cityscapes',
                    help='cityscapes, mapillary, camvid, kitti')

parser.add_argument('--num_workers', type=int, default=4,
                    help='cpu worker threads per dataloader instance')
parser.add_argument('--do_flip', action='store_true', default=False)
parser.add_argument('--edgeLoss', action='store_true', default=True,
                    help='edge weights')
parser.add_argument('--cv', type=int, default=0,
                    help=('Cross-validation split id to use. Default # of splits set'
                          ' to 3 in config'))
parser.add_argument('--full_crop_training', action='store_true', default=False,
                    help='Full Crop Training')
parser.add_argument('--pre_size', type=int, default=None,
                    help=('resize long edge of images to this before'
                          ' augmentation'))
parser.add_argument('--log_msinf_to_tb', action='store_true', default=False,
                    help='Log multi-scale Inference to Tensorboard')

parser.add_argument('--img_wt_loss', action='store_true', default=False,
                    help='per-image class-weighted loss')

parser.add_argument('--rescale', type=float, default=1.0,
                    help='Warm Restarts new lr ratio compared to original lr')
parser.add_argument('--repoly', type=float, default=1.5,
                    help='Warm Restart new poly exp')

parser.add_argument('--global_rank', default=0, type=int,
                    help='parameter used by apex library')

parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer')
parser.add_argument('--amsgrad', action='store_true', help='amsgrad for adam')

parser.add_argument('--test_mode', action='store_true', default=False,
                    help=('Minimum testing to verify nothing failed, '
                          'Runs code for 1 epoch of train and val'))

parser.add_argument('--trunk', type=str, default='resnet101',
                    help='trunk model, can be: resnet101 (default), resnet50')
parser.add_argument('--max_epoch', type=int, default=180)
parser.add_argument('--max_cu_epoch', type=int, default=150,
                    help='Class Uniform Max Epochs')
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--color_aug', type=float,
                    default=0.25, help='level of color augmentation')
parser.add_argument('--gblur', action='store_true', default=False,
                    help='Use Guassian Blur Augmentation')
parser.add_argument('--bblur', action='store_true', default=False,
                    help='Use Bilateral Blur Augmentation')
parser.add_argument('--brt_aug', action='store_true', default=False,
                    help='Use brightness augmentation')
parser.add_argument('--lr_schedule', type=str, default='poly',
                    help='name of lr schedule: poly')
parser.add_argument('--poly_exp', type=float, default=1.0,
                    help='polynomial LR exponent')
parser.add_argument('--poly_step', type=int, default=110,
                    help='polynomial epoch step')
parser.add_argument('--bs_trn', type=int, default=2,
                    help='Batch size for training per gpu')
parser.add_argument('--bs_val', type=int, default=2,
                    help='Batch size for Validation per gpu')
parser.add_argument('--crop_size', type=str, default='747',
                    help=('training crop size: either scalar or h,w'))
parser.add_argument('--scale_min', type=float, default=0.5,
                    help='dynamically scale training images down to this size')
parser.add_argument('--scale_max', type=float, default=2.0,
                    help='dynamically scale training images up to this size')
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--snapshot', type=str, default=None)
parser.add_argument('--resume', type=str, default=None,
                    help=('continue training from a checkpoint. weights, '
                          'optimizer, schedule are restored'))
parser.add_argument('--restore_optimizer', action='store_true', default=False)
parser.add_argument('--restore_net', action='store_true', default=False)
parser.add_argument('--exp', type=str, default='default',
                    help='experiment directory name')
parser.add_argument('--result_dir', type=str, default='./logs',
                    help='where to write log output')
parser.add_argument('--syncbn', action='store_true', default=False,
                    help='Use Synchronized BN')
parser.add_argument('--width', type=int, default=2200,
                    help='same size for all datasets')

parser.add_argument('--multiprocessing_distributed', action='store_true', default=False)
parser.add_argument('--dist_url', type=str, default="tcp://127.0.0.1:6789")
parser.add_argument('--dist_backend', type=str, default="nccl")
parser.add_argument('--world_size', type=int, default=1)
parser.add_argument('--rank', type=int, default=0)
parser.add_argument('--init_decoder', default=False, action='store_true',
                    help='initialize decoder with kaiming normal')

# Multi Scale Inference
parser.add_argument('--multi_scale_inference', action='store_true',
                    help='Run multi scale inference')

parser.add_argument('--default_scale', type=float, default=1.0,
                    help='default scale to run validation')


parser.add_argument('--eval', type=str, default=None,
                    help=('just run evaluation, can be set to val or trn or '
                          'folder'))
parser.add_argument('--eval_folder', type=str, default=None,
                    help='path to frames to evaluate')
parser.add_argument('--n_scales', type=str, default=None)

parser.add_argument('--sample_size', type=int, default=None,
                    help='sample size for fine-tuning')
parser.add_argument('--coarse_sample', type=int, default=None,
                    help='sample size for coarse')
parser.add_argument('--fine_sample', type=int, default=None,
                    help='sample size for fine')
parser.add_argument('--psl', type=int, default=None,
                    help='pseudo label iteration')
parser.add_argument('--backbone', type=str, default=None,
                    help='backbone')
parser.add_argument('--decoder', type=str, default=None,
                    help='decoder')
parser.add_argument('--max_iter', type=int, default=None,
                    help='max_itern')
parser.add_argument('--alpha', type=float, default=0.999,
                    help='max_itern')
parser.add_argument('--noEdge', action='store_true', default=False,
                    help='no edge loss')
parser.add_argument('--edge_wt', type=float, default=20,
                    help='max_itern')
parser.add_argument('--contrast_wt', type=float, default=0.1,
                    help='max_itern')
parser.add_argument('--c_inter', type=float, default=0.5,
                    help='inter domain')
parser.add_argument('--c_real', type=float, default=0.25,
                    help='real')
parser.add_argument('--c_syn', type=float, default=0.25,
                    help='synthetic')
parser.add_argument('--use_wl', action='store_true', default=False,
                    help='max_itern')
parser.add_argument('--not_ema', action='store_true', default=False,
                    help='max_itern')
parser.add_argument('--test', action='store_true', default=False,
                    help='max_itern')
parser.add_argument('--bn_buffer', action='store_true', default=False,
                    help='update bn buffers')
parser.add_argument('--use_contrast', action='store_true', default=False,
                    help='update bn buffers')
parser.add_argument('--pretrain', action='store_true', default=False,
                    help='update bn buffers')
parser.add_argument('--weak_label', type=str, default='point',
                    help='decoder')
parser.add_argument('--imloss', action='store_true', default=False,
                    help='use image loss')
parser.add_argument('--improto', action='store_true', default=False,
                    help='use image loss')
parser.add_argument('--synthia', action='store_true', default=False,
                    help='use image loss')

args = parser.parse_args()
args.best_record = {'epoch': -1, 'iter': 0, 'val_loss': 1e10, 'acc': 0,
                    'acc_cls': 0, 'mean_iu': 0, 'fwavacc': 0}

args.world_size = 1

# Test Mode run two epochs with a few iterations of training and val 
def find_free_port():
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes. /ptmp/andas/project/semantic-segmentation/logs/train_deepv3x71/full_synbn/code/train.py
    return port

def check_termination(epoch):
    if AutoResume:
        shouldterminate = AutoResume.termination_requested()
        if shouldterminate:
            if args.global_rank == 0:
                progress = "Progress %d%% (epoch %d of %d)" % (
                    (epoch * 100 / args.max_iter),
                    epoch,
                    args.max_iter
                )
                AutoResume.request_resume(
                    user_dict={"RESUME_FILE": logx.save_ckpt_fn,
                               "TENSORBOARD_DIR": args.result_dir,
                               "EPOCH": str(epoch)
                               }, message=progress)
                return 1
            else:
                return 1
    return 0

def main_process():
    return not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)

def get_module(module):
    if isinstance(module, torch.nn.parallel.DistributedDataParallel):
        return module.module
    return module

class ema_cls() :
    def __init__(self, net, args) :
        super(ema_cls, self).__init__()
        self.ema_model = net
        self.alpha = args.alpha
        self.buffer_keys = None
        self.init_ema_weights(net)

    def get_ema_model(self):
        return get_module(self.ema_model)

    def get_model(self, net) :
        return get_module(net)

    def _init_ema_weights(self, net, bn_buffer):
        for param in self.get_ema_model().parameters():
            param.detach_()
        mp = list(self.get_model(net).parameters())
        mcp = list(self.get_ema_model().parameters())
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()
        
        
    def _update_ema(self, net, itern, not_ema, bn_buffer):
        #print(self.alpha)
        alpha_teacher = min(1 - 1 / (itern + 1), self.alpha)
        for ema_param, param in zip(self.get_ema_model().parameters(),
                                    self.get_model(net).parameters()):
  
            if not param.data.shape:  # scalar tensor
                if not not_ema :
                    ema_param.data = \
                        alpha_teacher * ema_param.data + \
                        (1 - alpha_teacher) * param.data
                else :
                    ema_param.data = param.data
            else:
                if not not_ema :
                    ema_param.data[:] = \
                        alpha_teacher * ema_param[:].data[:] + \
                        (1 - alpha_teacher) * param[:].data[:]
                else : 
                    ema_param.data[:] = param[:].data[:]

class ema_cls_2() :
    def __init__(self, net, args) :
        super(ema_cls_2, self).__init__()
        self.ema_model = net
        self.alpha = args.alpha
        self.buffer_keys = None
        self._init_ema_weights(net)

    def get_ema_model(self):
        return get_module(self.ema_model)

    def get_model(self, net) :
        return get_module(net)

    def _init_ema_weights(self, net, bn_buffer=False):
        for param in self.get_ema_model().parameters():
            param.detach_()
        state_dict_main = self.get_model(net).state_dict()
        state_dict_ema = self.get_ema_model().state_dict()
        for (k_main, v_main), (k_ema, v_ema) in zip(state_dict_main.items(), state_dict_ema.items()):
            assert k_main == k_ema, "state_dict names are different!"
            assert v_main.shape == v_ema.shape, "state_dict shapes are different!"


    def _update_ema(self, net, itern, not_ema, bn_buffer):

        alpha_teacher = min(1 - 1 / (itern + 1), self.alpha)
        state_dict_main = self.get_model(net).state_dict()
        state_dict_ema = self.get_ema_model().state_dict()
        for (k_main, v_main), (k_ema, v_ema) in zip(state_dict_main.items(), state_dict_ema.items()):
            assert k_main == k_ema, "state_dict names are different!"
            assert v_main.shape == v_ema.shape, "state_dict shapes are different!"
            if 'num_batches_tracked' in k_ema or not_ema : 
                v_ema.copy_(v_main.clone().detach_())
            else:
                v_ema.copy_(v_ema * self.alpha + (1. - self.alpha) * v_main.clone().detach_())
            

def zipfolder(foldername, target_dir):            
    zipobj = zipfile.ZipFile(foldername + '.zip', 'w', zipfile.ZIP_DEFLATED)
    rootlen = len(target_dir) + 1
    for base, dirs, files in os.walk(target_dir):
        for file in files:
            fn = os.path.join(base, file)
            if "logs" not in fn :
                zipobj.write(fn, fn[rootlen:])

def main():
    """
    Main Function
    """
    if AutoResume:
        AutoResume.init()

    prep_experiment(args) 
    assert args.result_dir is not None, 'need to define result_dir arg' 
    ### fills args.ngpu 
  

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    args.ngpus_per_node = args.ngpu
    
    if args.ngpu == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False
    
    if args.multiprocessing_distributed:
        print("mulitprocessing done ")
        args.sync_bn = True
        args.train_gpu = [i for i in range(args.ngpu)]
        port = find_free_port()
        args.dist_url = f"tcp://127.0.0.1:{port}"
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else : 
        main_worker(args.train_gpu, args.ngpus_per_node, args)



def main_worker(gpu, ngpus_per_node, argss) :  
    global args
    args = argss
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
    
    print(gpu, ngpus_per_node, args.rank)

    logx.initialize(logdir=args.result_dir,
                    tensorboard=True, hparams=vars(args),
                    global_rank=args.global_rank)

    # Set up the Arguments, Tensorboard Writer, Dataloader, Loss Fn, Optimizer
    zipfolder(args.result_dir + '/code', '.') ### zip the current snapshot code in result dir
    assert_and_infer_cfg(args)
    
    train_loader_citi, train_loader_gta, val_loader, train_obj_citi, train_obj_gta = \
        datasets.setup_loaders(args)
    criterion, criterion_val = get_loss(args)

    if args.eval is None :
        citi_iter = iter(train_loader_citi)
        gta_iter = iter(train_loader_gta)


    auto_resume_details = None
    args.start_iter = 0

    if AutoResume:
        auto_resume_details = AutoResume.get_resume_details()

    proto = None
    if auto_resume_details:
        checkpoint_fn = auto_resume_details.get("RESUME_FILE", None)
        checkpoint = torch.load(checkpoint_fn,
                                map_location=torch.device('cpu'))
        args.result_dir = auto_resume_details.get("TENSORBOARD_DIR", None)
        args.start_epoch = int(auto_resume_details.get("EPOCH", None)) + 1
        args.restore_net = True
        args.restore_optimizer = True
        msg = ("Found details of a requested auto-resume: checkpoint={}"
               " tensorboard={} at epoch {}")
        logx.msg(msg.format(checkpoint_fn, args.result_dir,
                            args.start_epoch))
    elif args.resume:
        checkpoint = torch.load(args.resume,
                                map_location=torch.device('cpu'))
        #args.arch = checkpoint['arch']
        #args.start_epoch = int(checkpoint['epoch']) + 1
        args.start_iter = 0
        args.restore_net = True
        args.restore_optimizer = False
        msg = "Resuming from: checkpoint={}, epoch {}, arch {}"
        logx.msg(msg.format(args.resume, args.start_epoch, args.arch))
    elif args.snapshot:
        if 'ASSETS_PATH' in args.snapshot:
            args.snapshot = args.snapshot.replace('ASSETS_PATH', cfg.ASSETS_PATH)
        checkpoint = torch.load(args.snapshot,
                                map_location=torch.device('cpu'))
        args.restore_net = True
        msg = "Loading weights from: checkpoint={}".format(args.snapshot)
        logx.msg(msg)

    myfile = Path(args.result_dir + "/last_checkpoint.pth")

    if myfile.is_file() and args.snapshot is None :
        checkpoint = torch.load(args.result_dir + "/last_checkpoint.pth",
                                map_location=torch.device('cpu'))
        args.arch = checkpoint['arch']
        args.start_iter = int(checkpoint['iter']) + 1
        #args.start_epoch = int(args.start_epoch/len(train_loader))
        args.restore_net = True
        args.restore_optimizer = True
        proto = checkpoint['prototype'],
        msg = "Resuming from: checkpoint={}, epoch {}, arch {}"
        logx.msg(msg.format(args.resume, args.start_epoch, args.arch))

    net = network.get_net(args,  criterion)
    ema_net = network.get_net(args,  criterion)

    
    params_list = []
    params_list = group_weight_2(params_list, net, args.lr)

    optim = torch.optim.SGD(params_list,
                              lr=args.lr,
                              weight_decay=args.weight_decay,
                              momentum=args.momentum,
                              nesterov=False)
    
    contrast = ContrastCELoss(args.weak_label)
    contrast_intra = ContrastCELoss_intra(args.weak_label)
    imageloss = ImageLevelLoss()
    #iter_in_epoch = len(train_loader)
    #optim = torch.optim.AdamW(params_list, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)
    #total_iteration = int(args.max_epoch) *  iter_in_epoch + 1 #### update this, total images/batch_size
    if args.eval is None :
        total_iteration = args.max_iter 
        lr_policy = WarmUpPolyLR(args.lr, args.poly_exp, total_iteration, 1500) ### warmup iteration for deepv2 not warmup
    

    if args.restore_net:
        restore_net(net, ema_net, checkpoint)
    
    if args.distributed :
        torch.cuda.set_device(gpu)
        net = torch.nn.parallel.DistributedDataParallel(net.cuda(), device_ids=[gpu])
        ema_net = torch.nn.parallel.DistributedDataParallel(ema_net.cuda(), device_ids=[gpu])
    
    if args.restore_optimizer:
        restore_opt(optim, checkpoint)
    
    if args.init_decoder:
        net.module.init_mods()
    torch.cuda.empty_cache()

    #if args.start_epoch != 0:
    #    scheduler.step(args.start_epoch)

    # There are 4 options for evaluation:
    #  --eval val                           just run validation
    #  --eval val --dump_assets             dump all images and assets
    #  --eval folder                        just dump all basic images
    #  --eval folder --dump_assets          dump all images and assets
    if args.eval == 'val':
        validate(val_loader, net, ema_net, args.result_dir, criterion_val, optim, 0, None,
                     dump_assets=args.dump_assets,
                     dump_all_images=args.dump_all_images,
                     calc_metrics=not args.no_metrics)
        return 0
    elif args.eval == 'folder':
        # Using a folder for evaluation means to not calculate metrics
        validate(val_loader, net, args.result_dir, None, None, 0, None,
                 calc_metrics=False, dump_assets=args.dump_assets,
                 dump_all_images=True)
        return 0
    elif args.eval is not None:
        raise 'unknown eval option {}'.format(args.eval)

    train_main_loss = AverageMeter()
    train_edge_loss = AverageMeter()
    train_seg_sloss = AverageMeter()
    train_seg_mloss = AverageMeter()
    train_seg_uloss = AverageMeter()
    train_cont_loss = AverageMeter()
    train_im_loss = AverageMeter()
    train_cont_interloss = AverageMeter()
    train_cont_synloss = AverageMeter()
    train_cont_realloss = AverageMeter()

    train_main_loss.update(0,1)
    train_edge_loss.update(0,1)
    train_seg_sloss.update(0,1)
    train_seg_mloss.update(0,1)
    train_seg_uloss.update(0,1)
    train_cont_loss.update(0,1)
    train_im_loss.update(0,1)
    train_cont_interloss.update(0,1)
    train_cont_synloss.update(0,1)
    train_cont_realloss.update(0,1)

    start_time = None

    warmup_iter = 10 + args.start_iter
    
    epochs_since_start_citi = 0
    epochs_since_start_gta = 0

    if args.bn_buffer : 
        ema = ema_cls_2(ema_net, args)
    else :
        ema = ema_cls(ema_net, args)
    
    if args.pretrain :
        pretrain_iter = 30000
    else :
        pretrain_iter = -1

    if args.imloss :
        im_wt = 1.0
    else :
        im_wt = 0.0

    if args.weak_label == 'image' :
        type=0
    elif args.weak_label == 'point':
        type=1
    else :
        type=2

    if proto is not None :
        real_proto = proto[0]['real'].cuda()
        syn_proto = proto[0]['syn'].cuda()

        real_proto = prototypes(256, True, prto=real_proto, type=type) ### also need restore if restarting training
        syn_proto = prototypes(256, True, prto=syn_proto)
    else :
        real_proto = prototypes(256, True, type=type) 
        syn_proto = prototypes(256, True)

    for itern in range(args.start_iter, args.max_iter):    
        #validate(val_loader, net, net, args.result_dir, criterion_val, optim, 0)
        net.train()

        
        if itern == 0 :
            ema._init_ema_weights(net, args.bn_buffer)
        else :
            ema._update_ema(net, itern, args.not_ema, args.bn_buffer)
        
        
        if itern <= warmup_iter:
            start_time = time.time()

        try:
            batch_citi = next(citi_iter)
            if batch_citi[0].shape[0] != args.bs_trn:
                batch_citi = next(citi_iter)
        except:
            epochs_since_start_citi = epochs_since_start_citi + 1
            train_loader_citi.sampler.set_epoch(epochs_since_start_citi)
            update_epoch(epochs_since_start_citi)
            citi_iter = iter(train_loader_citi)
            batch_citi = next(citi_iter)
        try:
            batch_gta = next(gta_iter)
            if batch_gta[0].shape[0] != args.bs_trn:
                batch_gta = next(gta_iter)
        except:
            epochs_since_start_gta = epochs_since_start_gta + 1
            train_loader_gta.sampler.set_epoch(epochs_since_start_gta)
            gta_iter = iter(train_loader_gta)
            batch_gta = next(gta_iter)

        images_s, gts_s, edge_map_s, coarse_s, _img_name_s, scale_float_s = batch_gta
        images_u, gts_u, edge_map_u, coarse_u, _img_name_u, scale_float_u = batch_citi

        batch_pixel_size = images_u.size(0) * images_u.size(2) * images_u.size(3)
        images_u, gts_u, scale_float_u = images_u.cuda(), gts_u.cuda(), scale_float_u.cuda()
        inputs_u = {'images': images_u, 'gts': gts_u, 'coarse' : coarse_u}
        
        images_s, gts_s, scale_float_s = images_s.cuda(), gts_s.cuda(), scale_float_s.cuda()
        inputs_s = {'images': images_s, 'gts': gts_s, 'coarse' : coarse_s}

        optim.zero_grad()

        means = cfg.DATASET.MEAN
        means = [means for i in range(images_u.shape[0])]
        means = [torch.tensor(m) for m in means]
        means = torch.stack(means).view(-1, 3, 1, 1)
        std = cfg.DATASET.STD
        std = [std for i in range(images_u.shape[0])]
        std = [torch.tensor(m) for m in std]
        std = torch.stack(std).view(-1, 3, 1, 1)
        std = std.cuda()
        means = means.cuda()
        
        strong_parameters = {
            'mix': None,
            'color_jitter': random.uniform(0, 1),
            'blur': random.uniform(0, 1),
            'mean': means[0].unsqueeze(0),  # assume same normalization
            'std': std[0].unsqueeze(0)
        }

        features_u,logits_u,aspp_u, u_loss = net(inputs_u)
        if args.use_wl :
            seg_loss_u = u_loss['seg']
            seg_loss_u = seg_loss_u.mean() 
        
        classes_im = torch.zeros((gts_u.shape[0], 19))  ## for n_classes, all classes absent
        for i in range(gts_u.shape[0]) :
            cls = torch.unique(gts_u[i])
            cls = cls[cls != 255]  ### removing 255
            classes_im[i][cls] = 1 ## making present classes 0
        classes_im = classes_im.cuda()
        im_loss_u = imageloss(logits_u, classes_im)
        #im_loss_u = u_loss['image'] ### image loss common for all weak labels
        

        if not args.use_wl :
            gts_u_c = gts_u.clone()
            gts_u[gts_u > -1] = 255  #### if working only on image level label putting all labels as 255, not to be used by contrastive

        ## generate ps labels on citi images
        net_ema = ema.ema_model
        if itern > pretrain_iter or not args.use_wl: ## for use_wl case it looks upto pretrain iter
            features_s, logits_s, aspp, s_loss = net(inputs_s)
            pred_s = torch.nn.functional.softmax(logits_s, dim=1)
            pred_p, pred_s = pred_s.max(1)

            pred_s_c = torch.nn.functional.softmax(aspp, dim=1) ### for contrastive formulation in lower size
            pred_p_c, pred_s_c = pred_s_c.max(1)

            seg_loss_s = s_loss['seg']
            edge_loss_s = s_loss['edge']
            edge_loss_s = edge_loss_s.mean()
            seg_loss_s = seg_loss_s.mean()

            ema_features, ema_logits, aspp_ema, loss = net_ema(inputs_u)
            ema_softmax = torch.softmax(ema_logits.clone().detach(), dim=1)
            pseudo_prob, pseudo_label = torch.max(ema_softmax, dim=1)
            
            ema_softmax_c = torch.softmax(aspp_ema.clone().detach(), dim=1)
            pseudo_prob_c, pseudo_label_c = torch.max(ema_softmax_c, dim=1)

            #f1, logits1, aspp1, loss1 = net(inputs_u) ###################why again net(inputs_u) double here ... check
            ema1 = torch.softmax(aspp_u.clone().detach(), dim=1)
            prob1, label1 = torch.max(ema1, dim=1)

            c = None
            if args.use_wl :
                pseudo_label[gts_u != 255] = gts_u[gts_u != 255]                 ### important for cross domain augmentation, but not for contrastive loss
            
            ps_large_p = pseudo_prob.ge(0.968).long() == 1   ### can bring class balanced weight too here, but first make weak ps wt to 0
            ps_size = np.size(np.array(pseudo_label.cpu()))
            pseudo_weight = torch.sum(ps_large_p).item() / ps_size
            ps_wt = pseudo_weight

            wt_mask = torch.ones(pseudo_prob.shape) * ps_large_p.clone().cpu()
            pseudo_weight = pseudo_weight * wt_mask 
            pseudo_weight = pseudo_weight.cuda()         

            ############ computing the prototypes #################
            real_protoypes = real_proto.get_prototypes(ema_features, gts_u, pseudo_label_c, pseudo_prob_c, real=True) ### for real data
            syn_prototypes = syn_proto.get_prototypes(features_s, gts_s)
            ############## image loss wrt prototypes #######################
            
            ft_tes = features_u.clone()
            ft_tes = ft_tes.permute(0,2,3,1)
            logits = torch.matmul(ft_tes, torch.transpose(real_protoypes,0,1))
            logits = logits.permute(0,3,1,2)
            im_loss_pro = imageloss(logits, classes_im)
            if args.improto :
                protowt = 1.0
            else :
                protowt = 0.0

            if itern > 10000:
                cwt = 1.0
                im_wt = 1.0
            else:
                cwt = 0.0
                im_wt = 0.0


            im_loss_u_2 = protowt * im_loss_pro + im_loss_u

            prototype = {'real' : real_protoypes, 'syn' : syn_prototypes}
            
            if args.use_contrast : #and itern > 15000:
                contrast_loss_inter = contrast(features_s, gts_s, pred_s_c, pred_p_c, ema_features, gts_u, pseudo_label_c, pseudo_prob_c, real_protoypes)     ### use ema or student model for GT  features 
                contrast_loss_syn = contrast_intra(features_s, gts_s, pred_s_c, pred_p_c, syn_prototypes)
                #if itern > 10000 :
                contrast_loss_real = contrast_intra(features_u, gts_u, label1, prob1, real_protoypes)
                #else :
                #    contrast_loss_real = torch.tensor(0.0).cuda()


            pseudo_weight[:, :15, :] = 0
            #if psweight_ignore_bottom > 0:
            pseudo_weight[:, -120:, :] = 0
        
            gt_pixel_weight = torch.ones(pseudo_weight.shape)
            gt_pixel_weight = gt_pixel_weight.cuda()
        
            ### DACS mixing
            batch_size = args.bs_trn
            mixed_img, mixed_lbl = [None] * batch_size, [None] * batch_size
            mix_masks = get_class_masks(gts_s) ### mask from gta
        
            for i in range(batch_size):
                strong_parameters['mix'] = mix_masks[i]
                mixed_img[i], mixed_lbl[i] = strong_transform(
                    strong_parameters,
                    data=torch.stack((images_s[i], images_u[i])),
                    target=torch.stack((gts_s[i], pseudo_label[i])))
                _, pseudo_weight[i] = strong_transform(
                    strong_parameters,
                    target=torch.stack((gt_pixel_weight[i], pseudo_weight[i])))
            mixed_img = torch.cat(mixed_img)
            mixed_lbl = torch.cat(mixed_lbl)
            mixed_lbl = torch.squeeze(mixed_lbl,1)
            # Train on mixed images 
            inputs_mixed = {'images': mixed_img, 'gts': mixed_lbl, 'coarse' : None}
            mixed_features,_,_, mix_losses = net(inputs_mixed, pseudo_weight)
            seg_loss_m = mix_losses['seg']
            seg_loss_m = seg_loss_m.mean()

        edge_loss_s = args.edge_wt * edge_loss_s
        contrast_loss_inter = args.c_inter * contrast_loss_inter
        contrast_loss_real = args.c_real * contrast_loss_real
        contrast_loss_syn = args.c_syn * contrast_loss_syn
        
        contrast_loss = cwt * args.contrast_wt * (contrast_loss_inter +  contrast_loss_real +  contrast_loss_syn)
        
        im_loss_u_2 = im_wt * im_loss_u_2

        if args.distributed:
            if itern > pretrain_iter and args.use_wl :
                final_loss =  seg_loss_m + seg_loss_s + seg_loss_u + edge_loss_s + contrast_loss + im_loss_u_2
            elif not args.use_wl :
                final_loss =  seg_loss_m + seg_loss_s +  edge_loss_s + contrast_loss + im_loss_u_2
            elif args.use_wl :
                final_loss = seg_loss_u + im_loss_u_2

            log_main_loss = final_loss.clone().detach_()
            im_u = im_loss_u_2.clone().detach()

            if itern > pretrain_iter or not args.use_wl:
                edge_s = edge_loss_s.clone().detach_()
                seg_s = seg_loss_s.clone().detach_()
                seg_m = seg_loss_m.clone().detach_()
                if args.use_contrast : # and itern > 15000 :
                    cont = contrast_loss.clone().detach_()
                    torch.distributed.all_reduce(cont,
                                         torch.distributed.ReduceOp.SUM)
                    train_cont_loss.update(cont.item(), batch_pixel_size)
                    cont_inter = contrast_loss_inter.clone().detach_()
                    cont_syn = contrast_loss_syn.clone().detach_()
                    cont_real = contrast_loss_real.clone().detach_()
                    torch.distributed.all_reduce(cont_inter,
                                         torch.distributed.ReduceOp.SUM)
                    torch.distributed.all_reduce(cont_syn,
                                         torch.distributed.ReduceOp.SUM)
                    torch.distributed.all_reduce(cont_real,
                                         torch.distributed.ReduceOp.SUM)
                    train_cont_interloss.update(cont_inter.item(), batch_pixel_size)
                    train_cont_synloss.update(cont_syn.item(), batch_pixel_size)
                    train_cont_realloss.update(cont_real.item(), batch_pixel_size)

                torch.distributed.all_reduce(edge_s,
                                         torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(seg_s,
                                         torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(seg_m,
                                         torch.distributed.ReduceOp.SUM)
                train_edge_loss.update(edge_s.item(), batch_pixel_size)
                train_seg_sloss.update(seg_s.item(), batch_pixel_size)
                train_seg_mloss.update(seg_m.item(), batch_pixel_size)
                
            if args.use_wl :
                seg_u = seg_loss_u.clone().detach_()
                torch.distributed.all_reduce(seg_u,
                                             torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(log_main_loss,
                                         torch.distributed.ReduceOp.SUM)
            
            log_main_loss = log_main_loss / args.world_size
            if args.use_wl :
                train_seg_uloss.update(seg_u.item(), batch_pixel_size)
            
        else:
            main_loss = main_loss.mean()
            log_main_loss = main_loss.clone().detach_()

        train_main_loss.update(log_main_loss.item(), batch_pixel_size)
        train_im_loss.update(im_u.item(), batch_pixel_size)

        if torch.isnan(log_main_loss) : ### remove data before coninuing
            continue
        
        final_loss.backward()
        optim.step()

        current_idx = itern  #### niters_per_epoch 
        lr = lr_policy.get_lr(current_idx)

        for j in range(len(optim.param_groups)):
            if j > 1 :
                optim.param_groups[j]['lr'] = 10 * lr ### for head
            else :
                optim.param_groups[j]['lr'] = lr

        if itern >= warmup_iter:
            curr_time = time.time()
            batches = itern - warmup_iter + 1
            batchtime = (curr_time - start_time) / batches
        else:
            batchtime = 0

        curr_iter = current_idx
        
        if main_process() and curr_iter % 5 == 0:
            if itern > pretrain_iter and args.use_wl:
                msg = ('[iter {} / {}],  [train main loss {:0.6f}],'
                    '[seg synthetic loss {:0.6f}] [seg real loss {:0.6f}] [seg mixed loss {:0.6f}] [edgeloss {:0.6f}] [contloss {:0.6f}] [cont_inter {:0.6f}] [cont_syn {:0.6f}] [cont_real {:0.6f}]'
                    ' [lr {:0.6f}] [batchtime {:0.3g}]'
                    '[edgeloss instance {:0.6f}] [image loss {:0.6f}] [ps_wt {:0.6f}]'
                )
                msg = msg.format(
                    curr_iter, args.max_iter,  train_main_loss.avg, train_seg_sloss.avg, train_seg_uloss.avg, train_seg_mloss.avg, train_edge_loss.avg, train_cont_loss.avg, train_cont_interloss.avg, train_cont_synloss.avg, train_cont_realloss.avg,
                    optim.param_groups[-1]['lr'], batchtime, edge_s.item(), train_im_loss.avg, ps_wt)
                #logx.msg(msg)
            elif not args.use_wl:
                msg = ('[iter {} / {}],  [train main loss {:0.6f}],'
                    '[seg synthetic loss {:0.6f}] [seg mixed loss {:0.6f}] [edgeloss {:0.6f}] [contloss {:0.6f}] [cont_inter {:0.6f}] [cont_syn {:0.6f}] [cont_real {:0.6f}]'
                    ' [lr {:0.6f}] [batchtime {:0.9g}]'
                    '[edgeloss instance {:0.6f}] [image loss {:0.6f}] [ps_wt {:0.6f}]'
                )
                msg = msg.format(
                    curr_iter, args.max_iter,  train_main_loss.avg, train_seg_sloss.avg, train_seg_mloss.avg, train_edge_loss.avg, train_cont_loss.avg, train_cont_interloss.avg, train_cont_synloss.avg, train_cont_realloss.avg,
                    optim.param_groups[-1]['lr'], batchtime, edge_s.item(), train_im_loss.avg, ps_wt)
                #logx.msg(msg)
            else :
                msg = ('[iter {} / {}],  [train main loss {:0.6f}] [image loss {:0.6f}]'
                )
                msg = msg.format(
                    curr_iter, args.max_iter,  train_main_loss.avg, train_im_loss.avg)

            logx.msg(msg)

            if itern > pretrain_iter :
                metrics = {'loss': train_main_loss.avg,
                        'seg s loss' : train_seg_sloss.avg,
                        'seg m loss' : train_seg_mloss.avg,
                        'edge loss' : train_edge_loss.avg,
                        'lr': optim.param_groups[-1]['lr']}
            else :
                metrics = {'loss': train_main_loss.avg,
                    'seg s loss' : 0.0,
                    'seg m loss' : 0.0,
                    'edge loss' : 0.0,
                    'lr': optim.param_groups[-1]['lr']}

            logx.metric('train', metrics, curr_iter)

        if itern >= 10 and args.test_mode:
            del batch_citi, batch_gta
            return
        del batch_citi, batch_gta

        if curr_iter % 5000 == 0 and itern > 10:
            logx.msg("validating---------------")
            validate(val_loader, net, net_ema, args.result_dir, criterion_val, optim, curr_iter, prototype)
            net.train()
        
        if curr_iter % 100 == 0 and itern > 10:
            save_dict = {
            'iter': itern,
            'arch': args.arch,
            'num_classes': cfg.DATASET_INST.num_classes,
            'state_dict': net.state_dict(),
            'state_dict_ema' : net_ema.state_dict(),
            'optimizer': optim.state_dict(),
            'prototype' : prototype,
            }
            torch.save(save_dict, args.result_dir + "/last_checkpoint.pth")

        if check_termination(itern):
            return 0

def validate(val_loader, net, net_ema, result_dir, criterion, optim, epoch, prototype,
             calc_metrics=True,
             dump_assets=False,
             dump_all_images=False):
    """
    Run validation for one epoch 

    :val_loader: data loader for validation
    :net: the network
    :criterion: loss fn
    :optimizer: optimizer
    :epoch: current epoch
    :calc_metrics: calculate validation score
    :dump_assets: dump attention prediction(s) images
    :dump_all_images: dump all images, not just N
    """
    dumper = ImageDumper(val_len=len(val_loader),
                         dump_all_images=dump_all_images,
                         dump_assets=dump_assets,
                         dump_for_auto_labelling=False,
                         dump_for_submission=False)

    net.eval()
    val_loss = AverageMeter()
    iou_acc = 0

    for val_idx, data in enumerate(val_loader):
        input_images, labels, edge_map, coarse, img_names, _ = data 
        
        # Run network

        if 'SegFormer' in args.arch :
            assets, _iou_acc = \
                eval_minibatch_slide(data, net, criterion, val_loss, calc_metrics,
                          args, val_idx)
        else :
            assets, _iou_acc = \
                eval_minibatch(data, net, criterion, val_loss, calc_metrics,
                          args, val_idx)
        
        iou_acc += _iou_acc
 
        input_images, labels, edge_map, coarse, img_names, _ = data

        dumper.dump({'gt_images': labels,
                     'input_images': input_images,
                     'img_names': img_names,
                     'assets': assets}, val_idx)

        if main_process() : #saving the time
            if val_idx % 2 == 0:
                logx.msg(f'validating[Iter: {val_idx + 1} / {len(val_loader)}]')
        
    
    was_best = False
    if calc_metrics:
        was_best = eval_metrics(iou_acc, args, net, net_ema, optim, val_loss, epoch, result_dir, prototype)

    # Write out a summary html page and tensorboard image table
    
    dumper.write_summaries(was_best)


if __name__ == '__main__':
    main()
