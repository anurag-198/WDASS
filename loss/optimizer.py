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
"""

# Optimizer and scheduler related tasks

import math
import torch

from torch import optim
from runx.logx import logx
import sys 

from config import cfg
from loss.radam import RAdam
#from mmcv.runner import build_optimizer

def get_optimizer(args, net):
    """
    Decide Optimizer (Adam or SGD)
    """
    ''''
    cfg = dict(type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'pos_block': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'head': dict(lr_mult=10.)
                                                 }))

    optimizer = build_optimizer(net, cfg) ### need to check head in the network
    
    '''
    param_groups = []

    for k,v in net.named_parameters():
        if 'backbone' in k:
            param_groups.append({'params':[v], 'lr': args.lr})
        else:
            param_groups.append({'params':[v], 'lr': 10.0*args.lr})

    param_groups = net.parameters()

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(param_groups,
                              lr=args.lr,
                              weight_decay=args.weight_decay,
                              momentum=args.momentum,
                              nesterov=False)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(param_groups,
                               lr=args.lr,
                               weight_decay=args.weight_decay,
                               amsgrad=args.amsgrad)
    elif args.optimizer == 'radam':
        optimizer = RAdam(param_groups,
                          lr=args.lr,
                          weight_decay=args.weight_decay)
    else:
        raise ValueError('Not a valid optimizer')
    
    def poly_schd(epoch):
        return math.pow(1 - epoch / args.max_epoch, args.poly_exp)

    def poly2_schd(epoch):
        if epoch < args.poly_step:
            poly_exp = args.poly_exp
        else:
            poly_exp = 2 * args.poly_exp
        return math.pow(1 - epoch / args.max_epoch, poly_exp)

    if args.lr_schedule == 'scl-poly':
        if cfg.REDUCE_BORDER_EPOCH == -1:
            raise ValueError('ERROR Cannot Do Scale Poly')

        rescale_thresh = cfg.REDUCE_BORDER_EPOCH
        scale_value = args.rescale
        lambda1 = lambda epoch: \
             math.pow(1 - epoch / args.max_epoch,
                      args.poly_exp) if epoch < rescale_thresh else scale_value * math.pow(
                          1 - (epoch - rescale_thresh) / (args.max_epoch - rescale_thresh),
                          args.repoly)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    elif args.lr_schedule == 'poly2':
        scheduler = optim.lr_scheduler.LambdaLR(optimizer,
                                                lr_lambda=poly2_schd)
    elif args.lr_schedule == 'poly':
        scheduler = optim.lr_scheduler.LambdaLR(optimizer,
                                                lr_lambda=poly_schd)
    else:
        raise ValueError('unknown lr schedule {}'.format(args.lr_schedule))

    return optimizer, scheduler

def load_weights(net, optimizer, snapshot_file, restore_optimizer_bool=False):
    """
    Load weights from snapshot file
    """
    logx.msg("Loading weights from model {}".format(snapshot_file))
    net, optimizer = restore_snapshot(net, optimizer, snapshot_file, restore_optimizer_bool)
    return net, optimizer

def restore_snapshot(net, optimizer, snapshot, restore_optimizer_bool):
    """
    Restore weights and optimizer (if needed ) for resuming job.
    """
    checkpoint = torch.load(snapshot, map_location=torch.device('cpu'))
    logx.msg("Checkpoint Load Compelete")
    if optimizer is not None and 'optimizer' in checkpoint and restore_optimizer_bool:
        optimizer.load_state_dict(checkpoint['optimizer'])

    if 'state_dict' in checkpoint:
        net = forgiving_state_restore(net, checkpoint['state_dict'])
    else:
        net = forgiving_state_restore(net, checkpoint)

    return net, optimizer

def restore_opt(optimizer, checkpoint):
    assert 'optimizer' in checkpoint, 'cant find optimizer in checkpoint'
    optimizer.load_state_dict(checkpoint['optimizer'])

def test_state_dict(net, ema) :
    for k,v in net.items():
        if torch.ne(net[k],ema[k]).sum() == 0 :
            print("same ", k)
        else :
            print("not same", k)
    sys.exit()

def restore_net(net, ema_net, checkpoint):
    #test_state_dict(checkpoint['state_dict'], checkpoint['state_dict_ema'])
    if 'state_dict' in checkpoint :
        forgiving_state_restore(net, checkpoint['state_dict'])
    else :
        forgiving_state_restore(net, checkpoint)
    if 'state_dict_ema' in checkpoint :
        forgiving_state_restore(ema_net, checkpoint['state_dict_ema'])

def forgiving_state_restore(net, loaded_dict):
    """
    Handle partial loading when some tensors don't match up in size.
    Because we want to use models that were trained off a different
    number of classes.
    """

    net_state_dict = net.state_dict()
    new_loaded_dict = {}

    for k in net_state_dict:
        new_k = k
        if "module." + new_k in loaded_dict and net_state_dict[k].size() == loaded_dict["module." + new_k].size():
            new_loaded_dict[k] = loaded_dict["module." + new_k]
        elif new_k in loaded_dict and net_state_dict[k].size() == loaded_dict[new_k].size():
            new_loaded_dict[k] = loaded_dict[new_k]
        else:            
            logx.msg("Skipped loading parameter {}".format(k))
    net_state_dict.update(new_loaded_dict)
    net.load_state_dict(net_state_dict)
    return net
