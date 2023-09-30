import torch
import numpy as np
from abc import ABC

class prototypes(ABC) :
    def __init__(self, channels, weighted = True, alpha = 0.999, prto = None, type=-1): ### type = -1 is synthetic
        self.channels = channels
        self.weighted = weighted
        self.type=type

        if prto is None :
            self.prototypes = torch.zeros(19, self.channels).cuda()
        else :
            self.prototypes = prto
        self.alpha = alpha
        self.temperature = torch.tensor(1.0).cuda()

    def get_prototypes(self, features, gt=None, pseudo_label=None, pseudo_label_conf=None, real = True, img_lbl = False) : ### for img label dont send gt

        #### type-0 image label, depend on ps labels, type=1, pointlabel, center=point, type=2, center=average real features

        if self.type != 1 : ### for coarse, erroneous for point :( 
            gt = gt.unsqueeze(1).float().clone()
            gt =  torch.nn.functional.interpolate(gt, (features.shape[2], features.shape[3]), mode='nearest')
            gt = gt.squeeze(1).long()

        if pseudo_label is not None :
            pseudo_label[pseudo_label_conf < 0.95] = 255.0 ### labels unknown for low confident regions
            pseudo_label = pseudo_label.unsqueeze(1).float()
            pseudo_label_b = torch.nn.functional.interpolate(pseudo_label.clone(), (gt.shape[1], gt.shape[2]), mode='nearest')  
            gt = gt.unsqueeze(1).float()
            pseudo_label_b[gt!=255.0] = gt[gt!=255.0] ### to trust gt
            pseudo_label_b = torch.nn.functional.interpolate(pseudo_label_b, (pseudo_label.shape[2], pseudo_label.shape[3]), mode='nearest')
            pseudo_label = pseudo_label_b.long()
            pseudo_label = pseudo_label.squeeze(1)
            gt=gt.squeeze(1).long()
        else :
            pseudo_label = gt #### for the synthetic data, we do not send ps label, work with real features
        ### point label feature from interpolation #### different prototype creation for different data types ### todo later
        if self.type == 0 :
            gt = pseudo_label  ### for image label
        
        lbls = torch.unique(gt)

        ft = features.clone().detach() ### not require gradient on prototype features
        if self.type == 1 : ## only for point data ft to use for type = 1
            ft2 = ft.clone().detach()
            ft2 = torch.nn.functional.interpolate(ft, (gt.shape[1], gt.shape[2]), mode='bilinear')
            ft2 = ft2.permute(0,2,3,1)
        ft = ft.permute(0,2,3,1) 

        for i, index in enumerate(lbls):
            if index != 255 :
                if self.type == 1 : ### difficult for type = 1
                    mask = (gt == index).clone().detach()
                    fts = ft2[mask]
                    center = fts.sum(dim=0)/mask.sum()
                else :
                    mask = (gt == index).clone().detach() ## gt is ps label for image and actual gt wrt point and coarse
                    fts = ft[mask]
                    center = fts.sum(dim=0)/mask.sum() ### select wrt to real labels
                
                mask2 = (pseudo_label == index).clone().detach() ## for all labels(ps + GT)
                
                fts = ft[mask2]

                wt = torch.matmul(fts, center)
                wt = wt.view((-1,1))
                wt = wt/wt.sum()
                fts = wt * fts
                pt = fts.sum(dim=0)
                self.prototypes[index] = pt
        return self.prototypes

def main() :
    a = prototypes(256, True)
    ft = torch.randn((2,256,65,65)).cuda()
    lb = torch.randint(0,5,(2,65,65)).cuda()
    print("test123")
    pt = a.get_prototypes(ft, lb)
    print(pt.shape)

if __name__ == '__main__':
    main()