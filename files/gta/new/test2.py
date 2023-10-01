import numpy as np
import PIL.Image as Image
import sys
n = np.load("25000.npy")

for i in range(10188,n.shape[0])[::-1] :
    a = n[i]
    print("opening ", a)
    im = Image.open(a[0].strip()).convert('RGB')
    #im = Image.open("/BS/anurag/work/data/GTA_5/images/13558.png").convert('RGB')
    lb = Image.open(a[1].strip())

    im = np.array(im)
    lb = np.array(lb)

    if im.shape[0] != lb.shape[0] :
        print("bad ", a)
    
    print(im.shape, lb.shape) 
    if im.shape[1] != lb.shape[1] :
        print("bad ", a)

