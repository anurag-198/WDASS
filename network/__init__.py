"""
Network Initializations
"""

import importlib
import torch

from runx.logx import logx
from config import cfg
import torch.nn as nn


def get_net(args, criterion):
    """
    Get Network Architecture based on arguments provided
    """
    net = get_model(args, network='network.' + args.arch,
                    num_classes=cfg.DATASET.NUM_CLASSES,
                    criterion=criterion)
    if args.sync_bn :
        logx.msg("sync bn is initiated")
        net = nn.SyncBatchNorm.convert_sync_batchnorm(net)

    num_params = sum([param.nelement() for param in net.parameters()])
    logx.msg('Model params = {:2.1f}M'.format(num_params / 1000000))

    return net


def is_gscnn_arch(args):
    """
    Network is a GSCNN network
    """
    return 'gscnn' in args.arch


def wrap_network_in_dataparallel(net, gpu, use_apex_data_parallel=False):
    """
    Wrap the network in Dataparallel
    """
    print("wrapping in gpu ", gpu)
    if use_apex_data_parallel:
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[gpu])
        #net = apex.parallel.DistributedDataParallel(net)
    else:
        net = torch.nn.DataParallel(net)
    return net


def get_model(args, network, num_classes, criterion):
    """
    Fetch Network Function Pointer
    """
    module = network[:network.rfind('.')]
    model = network[network.rfind('.') + 1:]
    mod = importlib.import_module(module)
    net_func = getattr(mod, model)
    net = net_func(num_classes=num_classes, use_contrast = args.use_contrast, criterion=criterion)
    return net
