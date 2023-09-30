#!/usr/bin/env python3
# encoding: utf-8
# @Time    : 2018/9/28 下午12:13
# @Author  : yuchangqian
# @Contact : changqian_yu@163.com
# @File    : init_func.py.py
import torch
import torch.nn as nn
import sys

def __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum,
                  **kwargs):
    for name, m in feature.named_modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            conv_init(m.weight, **kwargs)
        elif isinstance(m, norm_layer):
            m.eps = bn_eps
            m.momentum = bn_momentum
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum,
                **kwargs):
    if isinstance(module_list, list):
        for feature in module_list:
            __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum,
                          **kwargs)
    else:
        __init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum,
                      **kwargs)


def group_weight_2(weight_group, module, lr) :
    bck_nd = []
    bck_d = []
    head_nd = []
    head_d = []
    for n,m in module.named_modules():
        if 'backbone' in n :
            if isinstance(m, nn.Linear):
                bck_d.append(m.weight)
                if m.bias is not None:
                    bck_nd.append(m.bias)
                
            elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
                bck_d.append(m.weight)
                if m.bias is not None:
                    bck_nd.append(m.bias)
                
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) \
                or isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm) or isinstance(m, nn.LayerNorm) or isinstance(m, nn.SyncBatchNorm):
                if m.weight is not None:
                     bck_nd.append(m.weight)
               
                if m.bias is not None:
                    bck_nd.append(m.bias)
                
            else:
                if hasattr(m, 'weight'):
                    if m.weight is not None :
                         bck_nd.append(m.weight)
                   
                elif hasattr(m, 'bias'):
                    if m.bias is not None:
                        bck_nd.append(m.bias)
                 
        else :
            if isinstance(m, nn.Linear):
                head_d.append(m.weight)
                if m.bias is not None:
                    head_nd.append(m.bias)
       
            elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
                head_d.append(m.weight)
                if m.bias is not None:
                    head_nd.append(m.bias)
          
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) \
                or isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm) or isinstance(m, nn.LayerNorm) or isinstance(m, nn.SyncBatchNorm):
                if m.weight is not None:
                     head_nd.append(m.weight)
           
                if m.bias is not None:
                    head_nd.append(m.bias)
              
            else:
                if hasattr(m, 'weight'):
                    if m.weight is not None :
                         head_nd.append(m.weight)
                
                elif hasattr(m, 'bias'):
                    if m.bias is not None:
                        head_nd.append(m.bias)
           


    #print(len(bck_d) + len(head_d) + len(head_nd) + len(bck_nd))
    #print(len(list(module.parameters())))
    #sys.exit()
    weight_group.append(dict(params=bck_d, lr=lr))
    weight_group.append(dict(params=bck_nd, lr=lr, weight_decay =.0))
    weight_group.append(dict(params=head_d, lr=10 * lr))
    weight_group.append(dict(params=head_nd, lr=10 * lr, weight_decay =.0))
    
    return weight_group

def group_weight(weight_group, module, lr):
    group_decay = []
    group_no_decay = []
    count = 0
    for n,m in module.named_modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
            else :
                print("none parameter ", n)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
            else :
                print("none parameter ", n)
        elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) \
                or isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm) or isinstance(m, nn.LayerNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            else :
                print("none parameter ", n)
            if m.bias is not None:
                group_no_decay.append(m.bias)
            else :
                print("none parameter ", n)
        else:
            if hasattr(m, 'weight'):
                if m.weight is not None :
                    group_no_decay.append(m.weight)
                else :
                    print("none parameter ", n)
            elif hasattr(m, 'bias'):
                if m.bias is not None:
                    group_no_decay.append(m.bias)
                else :
                    print("none parameter ", n)

    print(len(list(module.parameters())))
    #print(sum([param.nelement() for param in module.parameters()]))
    print(len([param for param in module.parameters()]))
    
    #print(sum([param.shape[0] for param in group_decay]))
    #print(sum([param.shape[0] for param in group_no_decay]))
    print(len(group_decay) + len(group_no_decay))

    assert len(list(module.parameters())) >= len(group_decay) + len(group_no_decay) ### always check for assertion
    weight_group.append(dict(params=group_decay, lr=lr))
    weight_group.append(dict(params=group_no_decay, weight_decay=.0, lr=lr))
    return weight_group