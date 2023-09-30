from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageLevelLoss2(nn.Module, ABC):
    def __init__(self, norm = False):
        super(ImageLevelLoss2, self).__init__()
        self.k = 1
        self.norm = norm  ### softmax prob of each class, normalise logits

    def forward(self, logits, gts) :
        
        if self.norm :
            logits = F.softmax(logits, dim = 1)
        logits_flat = logits.view((logits.shape[0], logits.shape[1], -1)) ### all the pixels in image
        logits_flat = self.k * logits_flat
        lse = torch.logsumexp(logits_flat, 2) ### pooling for whole logits 

        lse = (torch.log(torch.tensor(1/logits_flat.shape[-1])) + lse)/self.k
        pt = torch.sigmoid(lse)

        gts = gts.view((gts.shape[0], -1))
        classes = torch.zeros((gts.shape[0], 19))  ## for n_classes
        for i in range(gts.shape[0]) :
            cls = torch.unique(gts[i])
            cls = cls[cls != 255]  ### removing 255
            classes[i][cls] = 1
        classes = classes.cuda()
        loss = F.binary_cross_entropy(pt, classes)
        return loss
    
class ImageLevelLoss(nn.Module, ABC):
    def __init__(self, norm = False):
        super(ImageLevelLoss, self).__init__()
        self.k = 1
        self.norm = norm  ### softmax prob of each class, normalise logits

    def forward(self, logits, classes) :
        logits_flat = logits.view((logits.shape[0], logits.shape[1], -1)) ### all the pixels in image
        logits_flat = self.k * logits_flat
        lse = torch.logsumexp(logits_flat, 2) ### pooling for whole logits 
        lse = (torch.log(torch.tensor(1/logits_flat.shape[-1])) + lse)/self.k
        pt = torch.sigmoid(lse)
        loss = F.binary_cross_entropy(pt, classes)
        return loss
