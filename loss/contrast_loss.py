from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys

class PixelContrastLoss(nn.Module, ABC):
    def __init__(self, weak_label=None):
        super(PixelContrastLoss, self).__init__()

        self.temperature = 0.1
        self.base_temperature = 0.07

        self.ignore_label = 255

        self.max_samples = 512 ## 1024
        self.max_views = 20  # 100
        self.coarse = False
        if weak_label is not None and 'coarse' in weak_label :
            print("using coarse based contrastive ")
            self.coarse = True

    def _hard_anchor_sampling(self, X, y_hat, y, y_prob): #feat, labels, pred #f_fin, gts_fin, pred_fin, prob_fin
        batch_size, feat_dim = X.shape[0], X.shape[-1]
        classes = []
        total_classes = 0
        for ii in range(batch_size):
            this_y = y_hat[ii]
            this_classes = torch.unique(this_y)
            this_classes = [x for x in this_classes if x != self.ignore_label]
            this_classes = [x for x in this_classes if (this_y == x).nonzero().shape[0] > self.max_views]

            classes.append(this_classes)
            total_classes += len(this_classes)  ### just the classes in batch also > max_views = 100
        
        if total_classes == 0:
            return None, None

        n_view = self.max_samples // total_classes
        n_view = min(n_view, self.max_views)  ### lets say n_view = 100 (per class features)

        X_ = torch.zeros((total_classes, n_view, feat_dim), dtype=torch.float).cuda()
        y_ = torch.zeros(total_classes, dtype=torch.float).cuda()
        btch = torch.ones(total_classes, dtype=torch.float).cuda()

        X_ptr = 0
        for ii in range(batch_size):
            this_y_hat = y_hat[ii]
            this_y = y[ii]
            this_prob = y_prob[ii]

            this_classes = classes[ii] #### classes in an image

            for cls_id in this_classes:   
                hard_indices = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero()   ### lower limit of prediction
                easy_indices = ((this_y_hat == cls_id) & (this_y == cls_id) & (this_prob > 0.95)).nonzero()

                num_hard = hard_indices.shape[0]
                num_easy = easy_indices.shape[0]
                #print(num_easy, num_hard)
                if num_hard >= n_view :
                    num_hard_keep = n_view
                    num_easy_keep = n_view - num_hard_keep
                elif num_hard + num_easy >= n_view :
                    num_hard_keep = num_hard
                    num_easy_keep = n_view - num_hard_keep
                else:
                    continue
                
                if num_hard_keep > 0 :
                    perm = torch.randperm(num_hard)
                    hard_indices = hard_indices[perm[:num_hard_keep]]
                if num_easy_keep > 0 :
                    perm = torch.randperm(num_easy)
                    easy_indices = easy_indices[perm[:num_easy_keep]]
                
                if num_hard_keep == n_view :
                    indices = hard_indices
                elif num_easy_keep == n_view :
                    indices = easy_indices
                else :
                    indices = torch.cat((hard_indices, easy_indices), dim=0)

                X_[X_ptr, :, :] = X[ii, indices, :].squeeze(1)
                y_[X_ptr] = cls_id
                
                if ii < batch_size/2 :  ### to mark starting of real data
                    btch[X_ptr] = 0
                X_ptr += 1
        X_ = X_[:X_ptr]
        y_ = y_[:X_ptr]
        btch = btch[:X_ptr]
        return X_, y_, btch

    def _contrastive(self, feats_, labels_, btch, real_prototypes):
        anchor_num, n_view = feats_.shape[0], feats_.shape[1] ###anchor no : total anchor samples (total repetited classes in batch)
        pt = real_prototypes[labels_.long()]
        pt = pt.repeat_interleave(n_view, dim=0)
        pt = pt.view(feats_.shape)

        labels_ = labels_.contiguous().view(-1, 1)
        btch = btch.contiguous().view(-1,1)
        mask = torch.eq(labels_, torch.transpose(labels_, 0, 1)).float().cuda()
        mask2 = mask.clone()
        btch = torch.ne(btch, torch.transpose(btch, 0, 1)).float().cuda()
        contrast_count = n_view
        contrast_feature = torch.cat(torch.unbind(feats_, dim=1), dim=0)   ### total features (1,2,3,..7, 1,2,3..7, .....20 times)
        anchor_feature = contrast_feature
        anchor_count = contrast_count #### n_views (per class samples) 
        pt_features = torch.cat(torch.unbind(pt, dim=1), dim=0)

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, torch.transpose(pt_features, 0, 1)),
                                        self.temperature)   #### features x features (n_f x dim_f, initially) ##### important matrix size determiniation, no features to keep, large memory
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()  ### e ^ logits so more neg means more low val....

        mask = mask.repeat(anchor_count, contrast_count) #### x repeat, y repeat 
        btch = btch.repeat(anchor_count, contrast_count)
        
        neg_mask = 1 - mask

        #logits_mask = torch.ones_like(mask).scatter_(1, torch.arange(anchor_num * anchor_count).view(-1, 1).cuda(), 0) ### 0 in the diagonal 

        #mask = mask * logits_mask ### not including the diagonal all positives 
        mask = mask * btch  ### allow only cross domain positive pairs
        
        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True)

        exp_logits = torch.exp(logits)

        log_prob = logits - torch.log(exp_logits + neg_logits) ### n * n + n (for each row add negative sum correspondingly), logits because log(e^positivepair) is positivep
        
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)  ## again doing row wise per sample

        loss = -1 * (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()
        if torch.isnan(loss) :
            return torch.tensor(0.0).cuda()
        return loss

    def forward(self, features_s, gts_s, pred_s, pred_p, features_ema, gts_u, pseudo_label, pseudo_prob, real_prototypes) :
        gts_s = gts_s.unsqueeze(1).float().clone()
        gts_u = gts_u.unsqueeze(1).float().clone()
        pseudo_label = pseudo_label.unsqueeze(1).float().clone()

        gts_s = torch.nn.functional.interpolate(gts_s,        
                                                 (features_s.shape[2], features_s.shape[3]), mode='nearest')  
        
        pseudo_label_big = torch.nn.functional.interpolate(pseudo_label.clone(),        
                                                 (gts_u.shape[2], gts_u.shape[3]), mode='nearest')  
        
        pseudo_label_big[gts_u != 255] = gts_u[gts_u != 255]  ### to avoid gts from loosing inf with downsampling, all samples easy, use pseudo prob, only for points

        gts_u = torch.nn.functional.interpolate(gts_u,        
                                                 (features_s.shape[2], features_s.shape[3]), mode='nearest')  ### downsampling issue for point
        pseudo_label_big = torch.nn.functional.interpolate(pseudo_label_big,        
                                                 (features_s.shape[2], features_s.shape[3]), mode='nearest')  ### downsampling issue for point

        gts_s = gts_s.squeeze(1).long()
        pseudo_label_big = pseudo_label_big.squeeze(1).long()
        
        pseudo_label = pseudo_label.squeeze(1).long()
        gts_u = pseudo_label_big

        f_fin = torch.cat([features_s, features_ema], dim=0)
        gts_fin = torch.cat([gts_s, gts_u], dim=0)
        pred_fin = torch.cat([pred_s, pseudo_label], dim=0)
        prob_fin = torch.cat([pred_p, pseudo_prob], dim=0)
        
        #f_fin = F.normalize(f_fin, dim=1)  ### in a unit square
        
        gts_fin = gts_fin.squeeze(1).long()
        pred_fin = pred_fin.squeeze(1).long()
        prob_fin = prob_fin.squeeze(1)

        assert gts_fin.shape[-1] == f_fin.shape[-1], '{} {}'.format(gts_fin.shape, f_fin.shape)

        batch_size = f_fin.shape[0]

        gts_fin = gts_fin.contiguous().view(batch_size, -1)
        pred_fin = pred_fin.contiguous().view(batch_size, -1)
        prob_fin = prob_fin.contiguous().view(batch_size, -1)

        f_fin = f_fin.permute(0, 2, 3, 1)
        f_fin = f_fin.contiguous().view(f_fin.shape[0], -1, f_fin.shape[-1])

        feats_, labels_, btch = self._hard_anchor_sampling(f_fin, gts_fin, pred_fin, prob_fin)
        if feats_ is None :
            return torch.tensor(0.0).cuda()
        loss = self._contrastive(feats_, labels_, btch, real_prototypes)
        return loss

class ContrastCELoss(nn.Module, ABC):
    def __init__(self, weak_label=None):
        super(ContrastCELoss, self).__init__()
        ignore_index = 255
        self.contrast_criterion = PixelContrastLoss(weak_label)

    def forward(self,features_s, gts_s, pred_s, pred_p, features_ema, gts_u, pseudo_label, pseudo_prob, real_prototypes): ### features, gt (features_s, gts_s, features_u, pseudo_label, gts_u)
        loss_contrast = self.contrast_criterion(features_s, gts_s, pred_s, pred_p, features_ema, gts_u, pseudo_label, pseudo_prob, real_prototypes)
        return loss_contrast