"""
Code Adapted from:
https://github.com/sthalles/deeplab_v3

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
import torch
from torch import nn

from network.mynn import initialize_weights, Norm2d, Upsample
from network.utils import get_aspp, get_trunk, make_seg_head
import torch.nn.functional as F

class DeepV3Plus(nn.Module):
    """
    DeepLabV3+ with various trunks supported
    Always stride8
    """
    def __init__(self, num_classes, trunk='wrn38', criterion=None,
                 use_dpc=False, init_all=False):
        super(DeepV3Plus, self).__init__()
        self.criterion = criterion
        self.backbone, s2_ch, _s4_ch, high_level_ch = get_trunk(trunk)
        self.aspp, aspp_out_ch = get_aspp(high_level_ch,
                                          bottleneck_ch=256,
                                          output_stride=8,
                                          dpc=use_dpc)
        self.bot_fine = nn.Conv2d(s2_ch, 48, kernel_size=1, bias=False)
        self.bot_aspp = nn.Conv2d(aspp_out_ch, 256, kernel_size=1, bias=False)
        self.final = nn.Sequential(
            nn.Conv2d(256 + 48, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1, bias=False))

        if init_all:
            initialize_weights(self.aspp)
            initialize_weights(self.bot_aspp)
            initialize_weights(self.bot_fine)
            initialize_weights(self.final)
        else:
            initialize_weights(self.final)

    def forward(self, inputs):
        assert 'images' in inputs
        x = inputs['images']

        x_size = x.size()
        s2_features, _, final_features = self.backbone(x)
        aspp = self.aspp(final_features)
        conv_aspp = self.bot_aspp(aspp)
        conv_s2 = self.bot_fine(s2_features)
        conv_aspp = Upsample(conv_aspp, s2_features.size()[2:])
        cat_s4 = [conv_s2, conv_aspp]
        cat_s4 = torch.cat(cat_s4, 1)
        final = self.final(cat_s4)
        out = Upsample(final, x_size[2:])

        if self.training:
            assert 'gts' in inputs
            gts = inputs['gts']
            coarse = inputs['coarse']
            return out, self.criterion(out, gts, coarse)
            #return self.criterion(out, gts)

        return {'pred': out}


def DeepV3PlusSRNX50(num_classes, criterion):
    return DeepV3Plus(num_classes, trunk='seresnext-50', criterion=criterion)


def DeepV3PlusR50(num_classes, criterion):
    return DeepV3Plus(num_classes, trunk='resnet-50', criterion=criterion)


def DeepV3PlusSRNX101(num_classes, criterion):
    return DeepV3Plus(num_classes, trunk='seresnext-101', criterion=criterion)


def DeepV3PlusW38(num_classes, criterion):
    return DeepV3Plus(num_classes, trunk='wrn38', criterion=criterion)


def DeepV3PlusW38I(num_classes, criterion):
    return DeepV3Plus(num_classes, trunk='wrn38', criterion=criterion,
                      init_all=True)


def DeepV3PlusX71(num_classes, criterion):
    return DeepV3Plus(num_classes, trunk='xception71', criterion=criterion)


def DeepV3PlusEffB4(num_classes, criterion):
    return DeepV3Plus(num_classes, trunk='efficientnet_b4',
                      criterion=criterion)


class DeepV2(nn.Module):
    """
    DeepLabV2 with various trunks supported
    """
    def __init__(self, num_classes, use_contrast, trunk='resnet-50', criterion=None,
                 use_dpc=False, init_all=False, output_stride=8):
        super(DeepV2, self).__init__()
        self.criterion = criterion

        self.backbone, _s2_ch, _s4_ch, high_level_ch = \
            get_trunk(trunk, output_stride=output_stride)
        self.aspp = get_aspp(high_level_ch,
                                          bottleneck_ch=num_classes,
                                          output_stride=output_stride,
                                          dpc=use_dpc)
        self.representation = nn.Sequential(nn.Conv2d(2048, 256, 1))
        self.contrast = use_contrast

        initialize_weights(self.aspp)

    def forward(self, inputs, wt = None):
        assert 'images' in inputs
        x = inputs['images']
        x_size = x.size()
        
        _, _, final_features = self.backbone(x)
        aspp = self.aspp(final_features)
        out = Upsample(aspp, x_size[2:])
        final_feat = self.representation(final_features)
        final_feat = F.normalize(final_feat, dim=1)
        
        if self.training:
            assert 'gts' in inputs
            gts = inputs['gts']
            coarse = None
            if inputs['coarse'] is not None :
                coarse = inputs['coarse']
            if self.contrast :
                return final_feat, out, aspp, self.criterion(out, gts, coarse, wt, final_features)
            else :
                return out, self.criterion(out, gts, coarse, wt)
        return {'pred': out}

def DeepV2R101(num_classes, use_contrast, criterion) :
    return DeepV2(num_classes, use_contrast, trunk='resnet-101', criterion=criterion)