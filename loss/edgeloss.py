"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

# Code adapted from:
# https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb
#
# MIT License
#
# Copyright (c) 2016 Eric Jang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from .contrast_loss import ContrastCELoss
from .image_label import ImageLevelLoss
def calc_pad_same(in_siz, out_siz, stride, ksize):
    """Calculate same padding width.
    Args:
    ksize: kernel size [I, J].
    Returns:
    pad_: Actual padding width.
    """
    return (out_siz - 1) * stride + ksize - in_siz


def conv2d_same(input, kernel, groups,bias=None,stride=1,padding=0,dilation=1):
    n, c, h, w = input.shape
    kout, ki_c_g, kh, kw = kernel.shape
    pw = calc_pad_same(w, w, 1, kw)
    ph = calc_pad_same(h, h, 1, kh)
    pw_l = pw // 2
    pw_r = pw - pw_l
    ph_t = ph // 2
    ph_b = ph - ph_t

    input_ = F.pad(input, (pw_l, pw_r, ph_t, ph_b))
    result = F.conv2d(input_, kernel, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
    assert result.shape == input.shape
    return result

def gradient_central_diff(input, cuda):
    #return input, input
    kernel = [[1, 0, -1]]
    kernel_t = 0.5 * torch.Tensor(kernel) * -1.  # pytorch implements correlation instead of conv
    if type(cuda) is int:
        if cuda != -1:
            kernel_t = kernel_t.cuda(device=cuda)
    else:
        if cuda is True:
            kernel_t = kernel_t.cuda()
    n, c, h, w = input.shape

    x = conv2d_same(input, kernel_t.unsqueeze(0).unsqueeze(0).repeat([c, 1, 1, 1]), c)
    y = conv2d_same(input, kernel_t.t().unsqueeze(0).unsqueeze(0).repeat([c, 1, 1, 1]), c)
    return x, y

def compute_single_sided_diferences(o_x, o_y, input):
    # n,c,h,w
    #input = input.clone()
    o_y[:, :, 0, :] = input[:, :, 1, :].clone() - input[:, :, 0, :].clone()
    o_x[:, :, :, 0] = input[:, :, :, 1].clone() - input[:, :, :, 0].clone()
    # --
    o_y[:, :, -1, :] = input[:, :, -1, :].clone() - input[:, :, -2, :].clone()
    o_x[:, :, :, -1] = input[:, :, :, -1].clone() - input[:, :, :, -2].clone()
    return o_x, o_y

def numerical_gradients_2d(input, cuda=False):
    """
    numerical gradients implementation over batches using torch group conv operator.
    the single sided differences are re-computed later.
    it matches np.gradient(image) with the difference than here output=x,y for an image while there output=y,x
    :param input: N,C,H,W
    :param cuda: whether or not use cuda
    :return: X,Y
    """
    n, c, h, w = input.shape
    assert h > 1 and w > 1
    x, y = gradient_central_diff(input, cuda)
    return x, y

def convTri(input, r, cuda=False):
    """
    Convolves an image by a 2D triangle filter (the 1D triangle filter f is
    [1:r r+1 r:-1:1]/(r+1)^2, the 2D version is simply conv2(f,f'))
    :param input:
    :param r: integer filter radius
    :param cuda: move the kernel to gpu
    :return:
    """
    if (r <= 1):
        raise ValueError()
    n, c, h, w = input.shape
    return input
    f = list(range(1, r + 1)) + [r + 1] + list(reversed(range(1, r + 1)))
    kernel = torch.Tensor([f]) / (r + 1) ** 2
    if type(cuda) is int:
        if cuda != -1:
            kernel = kernel.cuda(device=cuda)
    else:
        if cuda is True:
            kernel = kernel.cuda()

    # padding w
    input_ = F.pad(input, (1, 1, 0, 0), mode='replicate')
    input_ = F.pad(input_, (r, r, 0, 0), mode='reflect')
    input_ = [input_[:, :, :, :r], input, input_[:, :, :, -r:]]
    input_ = torch.cat(input_, 3)
    t = input_

    # padding h
    input_ = F.pad(input_, (0, 0, 1, 1), mode='replicate')
    input_ = F.pad(input_, (0, 0, r, r), mode='reflect')
    input_ = [input_[:, :, :r, :], t, input_[:, :, -r:, :]]
    input_ = torch.cat(input_, 2)

    output = F.conv2d(input_,
                      kernel.unsqueeze(0).unsqueeze(0).repeat([c, 1, 1, 1]),
                      padding=0, groups=c)
    output = F.conv2d(output,
                      kernel.t().unsqueeze(0).unsqueeze(0).repeat([c, 1, 1, 1]),
                      padding=0, groups=c)
    return output

def compute_normal(E, cuda=False):
    if torch.sum(torch.isnan(E)) != 0:
        print('nans found here')
        import ipdb;
        ipdb.set_trace()
    E_ = convTri(E, 4, cuda)
    Ox, Oy = numerical_gradients_2d(E_, cuda)
    Oxx, _ = numerical_gradients_2d(Ox, cuda)
    Oxy, Oyy = numerical_gradients_2d(Oy, cuda)

    aa = Oyy * torch.sign(-(Oxy + 1e-5)) / (Oxx + 1e-5)
    t = torch.atan(aa)
    O = torch.remainder(t, np.pi)

    if torch.sum(torch.isnan(O)) != 0:
        print('nans found here')
        import ipdb;
        ipdb.set_trace()

    return O

def compute_normal_2(E, cuda=False):
    if torch.sum(torch.isnan(E)) != 0:
        print('nans found here')
        import ipdb;
        ipdb.set_trace()
    E_ = convTri(E, 4, cuda)
    Ox, Oy = numerical_gradients_2d(E_, cuda)
    Oxx, _ = numerical_gradients_2d(Ox, cuda)
    Oxy, Oyy = numerical_gradients_2d(Oy, cuda)

    aa = Oyy * torch.sign(-(Oxy + 1e-5)) / (Oxx + 1e-5)
    t = torch.atan(aa)
    O = torch.remainder(t, np.pi)

    if torch.sum(torch.isnan(O)) != 0:
        print('nans found here')
        import ipdb;
        ipdb.set_trace()

    return O, (Oyy, Oxx)

def compute_grad_mag(E, cuda=False):
    E_ = convTri(E, 4, cuda)
    Ox, Oy = numerical_gradients_2d(E_, cuda)
    mag = torch.sqrt(torch.mul(Ox,Ox) + torch.mul(Oy,Oy) + 1e-6)
    mag = mag / mag.max();
    return mag

def perturbate_input_(input, n_elements=200):
    N, C, H, W = input.shape
    assert N == 1
    c_ = np.random.random_integers(0, C - 1, n_elements)
    h_ = np.random.random_integers(0, H - 1, n_elements)
    w_ = np.random.random_integers(0, W - 1, n_elements)
    for c_idx in c_:
        for h_idx in h_:
            for w_idx in w_:
                input[0, c_idx, h_idx, w_idx] = 1
    return input

def _sample_gumbel(shape, eps=1e-10):
    """
    Sample from Gumbel(0, 1)

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    U = torch.rand(shape).cuda()
    return - torch.log(eps - torch.log(U + eps))

def _gumbel_softmax_sample(logits, tau=1, eps=1e-10):
    """
    Draw a sample from the Gumbel-Softmax distribution

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb
    (MIT license)
    """
    assert logits.dim() == 3
   
    gumbel_noise = _sample_gumbel(logits.size(), eps=eps)
    y = logits + gumbel_noise
    return F.softmax(y / tau, 1)

def _one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """

    y = torch.eye(num_classes).cuda()
    return y[labels].permute(0,3,1,2)

class EdgeLoss(nn.Module):
    def __init__(self, weight=None, noEdge = False, use_contrast = False, cuda=True, ignore_index=255,
                 reduction='none'):
        super(EdgeLoss, self).__init__()
        self._cuda = cuda
        self.nll_loss = nn.NLLLoss(weight, reduction=reduction,
                                   ignore_index=ignore_index)
        self.noEdge = noEdge
        self.use_contrast = use_contrast
        self.contrast = ContrastCELoss()
        self.imloss = ImageLevelLoss()
        return

    def forward(self, input_logits, gts,  coarse=None, wt=None, features=None, ignore_pixel=255):
        """
        :param input_logits: NxCxHxW
        :param gt_semantic_masks: NxCxHxW
        :return: final loss
        """
        losses = {}
        seg_loss = self.nll_loss(F.log_softmax(input_logits, dim=1), gts)
        #print("logits-----------", input_logits.shape, gts.shape)
        
        if wt is not None  :
            seg_loss = wt * seg_loss

        seg_loss_f = seg_loss[gts != 255]
        seg_loss_f = seg_loss_f.mean()
        
        edgeloss = torch.tensor(0.0).cuda()

        if coarse is None or self.noEdge :
            losses = {}
            losses['seg'] = seg_loss_f
            losses['edge'] = edgeloss
            return losses

        #if torch.sum(coarse) > 0 :  ### only on weak labels
            #gts_wl = gts.clone()
            #input_logits_wl = input_logits.clone()
            #gts_wl = gts_wl[coarse]
            #input_logits_wl = input_logits_wl[coarse]
            #wl_loss = self.imloss(input_logits_wl, gts_wl)
            #losses['image'] = wl_loss

        if torch.sum(torch.logical_not(coarse)) > 0 :
            gts_ed = gts.clone()
            input_logits_ed = input_logits.clone()

            gts_ed = gts_ed[torch.logical_not(coarse)]
            input_logits_ed = input_logits_ed[torch.logical_not(coarse)]

            N, C, H, W = input_logits_ed.shape
            th = 1e-8  # 1e-10
            eps = 1e-10
            ignore_mask = (gts_ed == ignore_pixel).detach()
            input_logits_ed = torch.where(ignore_mask.view(N, 1, H, W).expand(N, 19, H, W),
                                   torch.zeros(N,C,H,W).cuda(),
                                   input_logits_ed)
            gt_semantic_masks = gts_ed.detach()
            gt_semantic_masks = torch.where(ignore_mask, torch.zeros(N,H,W).long().cuda(), gt_semantic_masks) ### not perfect but will work [selective ones can be difficult]
            gt_semantic_masks = _one_hot_embedding(gt_semantic_masks, 19).detach()

            g = _gumbel_softmax_sample(input_logits_ed.view(N, C, -1), tau=0.5)
            g = g.reshape((N, C, H, W))
            g = compute_grad_mag(g, cuda=self._cuda)
 
            g_hat = compute_grad_mag(gt_semantic_masks, cuda=self._cuda)
            g = g.view(N, -1)
            g_hat = g_hat.contiguous().view(N, -1)
            loss_ewise = F.l1_loss(g, g_hat, reduction='none', reduce=False)

            p_plus_g_mask = (g >= th).detach().float()
            loss_p_plus_g = torch.sum(loss_ewise * p_plus_g_mask) / (torch.sum(p_plus_g_mask) + eps)

            p_plus_g_hat_mask = (g_hat >= th).detach().float()
            loss_p_plus_g_hat = torch.sum(loss_ewise * p_plus_g_hat_mask) / (torch.sum(p_plus_g_hat_mask) + eps)

            total_loss = 0.5 * loss_p_plus_g + 0.5 * loss_p_plus_g_hat
            edgeloss = total_loss
        
        losses['seg'] = seg_loss_f
        losses['edge'] = edgeloss
        return losses
