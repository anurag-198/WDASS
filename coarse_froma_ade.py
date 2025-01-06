import json
import sys
import numpy as np
import pathlib
import shutil
import cv2 as cv
from PIL import Image
from collections import namedtuple
import random
import PIL
import PIL.Image
import os

from matplotlib.path import Path

from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage import label

from tabulate import tabulate

files = [] 

dir = "1/2/3/data.txt"



Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category leve
    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )


#--------------------------------------------------------------------------------
# A list of all labels
#--------------------------------------------------------------------------------

# Please adapt the train IDs as appropriate for you approach.
# Note that you might want to ignore labels with ID 255 during training.
# Further note that the current train IDs are only a suggestion. You can use whatever you like.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!

labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
]


#--------------------------------------------------------------------------------
# Create dictionaries for a fast lookup
#--------------------------------------------------------------------------------

# Please refer to the main method below for example usages!

# name to label object
name2label      = { label.name    : label for label in labels           }
# id to label object
id2label        = { label.id      : label for label in labels           }
# trainId to label object
trainId2label   = { label.trainId : label.id for label in reversed(labels) }
# label2trainid
label2trainid   = { label.id      : label.trainId for label in labels   }
# trainId to label object
trainId2name   = { label.trainId : label.name for label in labels   }
trainId2color  = { label.trainId : label.color for label in labels      }


palette=[[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
                 [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
                 [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
                 [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
                 [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
                 [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
                 [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
                 [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
                 [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
                 [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
                 [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
                 [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
                 [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
                 [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
                 [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
                 [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
                 [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
                 [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
                 [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
                 [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
                 [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
                 [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
                 [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
                 [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
                 [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
                 [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
                 [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
                 [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
                 [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
                 [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
                 [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
                 [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
                 [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
                 [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
                 [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
                 [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
                 [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
                 [102, 255, 0], [92, 0, 255]]

pal = []
for p in palette:
    pal.append(p[0])
    pal.append(p[1])
    pal.append(p[2])

zero_pad = 256 * 3 - len(pal)

for i in range(zero_pad):
    pal.append(0)

color_mapping = pal

def colorize_mask(image_array):
    """
    Colorize the segmentation mask
    """
    new_mask = Image.fromarray(image_array.astype(np.uint8)).convert('P')
    new_mask.putpalette(color_mapping)
    return new_mask

# find adaptive k, which results in around 35% erosion
def do_erosion(mask):
    n_labels = np.sum(mask > 0)
    k = int(np.sqrt(n_labels))
    lastpcs = []
    lasthighk = n_labels/2
    lastlowk = 0
    bestdiff = 1
    mask2 = mask
    cnt = 0

    flag = True
    while(flag) :
        kernel = np.ones((k,k),np.uint8)
        mask2 = cv.erode(mask, kernel,iterations = 1)
        pc = 1-(np.sum(mask2 > 0)/n_labels)
        lastpcs.append(pc)
        diff = 0.35 - pc
        if diff > 0 : #### do more erosion, increase k
            lastlowk = k
            k = k + int((lasthighk - k)/2)
        else:
            lasthighk = k
            k = k - int((k - lastlowk)/2)
        if(lasthighk - lastlowk < 3) :
            flag = False
    return mask2

### small -> if smaller than 1/100th of image
def performeros(mask) :
    h,w = mask.shape
    n_labels = np.sum(mask > 0)
    if n_labels < (h*w)/1000 :
        mask[mask > 0] = 0
        return mask
    #### obtain erosion factor ####
    mask = do_erosion(mask)
    return mask

def dis(a,b) :
    return np.sqrt((a[0]-b[0])**2 + (a[1] - b[1])**2)

def findclosest(x, li) :
    h = 1000000
    sol = None

    for el in li :
        if dis(x,el) < h : 
            h = dis(x,el)
            print(x, el, h)
            sol = el
    return sol

def getlinepts(a,b) :
    x_mn = min(a[0],b[0])
    x_mx = max(a[0],b[0])

    y_mn = min(a[1],b[1])
    y_mx = max(a[1], b[1])

    val = []
    val.append(a)
    val.append(b)
    #### this is done wrt to x, need to also do wrt y
    m = (b[1] - a[1])/(b[0] - a[0])
    for c in range(x_mn+1,x_mx):
        y = int(m * (c - a[0])) + a[1]
        val.append([c, y])
    for c in range(y_mn+1,y_mx):
        x = int((1/m) * (c - a[1])) + a[0]
        val.append([x, c])
    
    return val

def get_neighbor(pt, apts) :
    a,b = pt
    if tuple((a,b-1)) in apts :
        return tuple((a, b-1))
    if tuple((a+1,b-1)) in apts :
        return tuple((a+1, b-1))
    if tuple((a+1,b)) in apts :
        return tuple((a+1, b))
    if tuple((a+1,b+1)) in apts :
        return tuple((a+1, b+1))
    if tuple((a,b+1)) in apts :
        return tuple((a, b+1))
    if tuple((a-1,b+1)) in apts :
        return tuple((a-1, b+1))
    if tuple((a-1,b)) in apts :
        return tuple((a-1, b))
    if tuple((a-1,b-1)) in apts :
        return tuple((a-1, b-1))
    return None

def converttoc(pts) :
    pt = [tuple((v[1],v[0])) for v in pts]
    return pt

def getmask(polypts, mp) :
    x, y = np.meshgrid(np.arange(mp.shape[1]), np.arange(mp.shape[0])) # make a canvas with coordinates(x,y), shape[0] -- rows
    x, y = x.flatten(), y.flatten()
    points = np.vstack((x,y)).T
    ## in coordinate system ##
    pts = converttoc(polypts)
    p = Path(pts) # make a polygon
    grid = p.contains_points(points)
    mask = grid.reshape(mp.shape[0],mp.shape[1])
    return mask

### retrive the polygon mp is the edge, pts is the points on the edge
def getpoly(pts, mp) :
    ed = np.zeros(mp.shape)
    ### first see 1/50 sampling
    sep = int(pts.shape[0]/50)
    cnt = 0
    cnt2 = 0
    #print(pts.shape)
    #sys.exit()
    allpts = [tuple((v[0], v[1])) for v in pts]
    flag2 = True
    while(flag2) :
        polypts = []
        fp = []
        dg = False
        flag = True
        if not allpts :
            break 
        lastkey = allpts[0]
        key = allpts[0]
        #print("key----->", key)
        firstpt = allpts[0]
        polypts.append(firstpt)   
        while(flag) :
            allpts.remove(key)
            ## walk along the neighbors till separator
            key = get_neighbor(key, allpts)
            if key is None :
                dg = True
            cnt2 += 1
            cnt += 1
            #print(cnt, cnt2)
            if cnt == sep or key is None:
                #print("key---sep--", key)
                #print("allpts----", allpts)
                if key is None :
                    key = firstpt
                polypts.append(key)
                #import pdb; pdb.set_trace()
                pt_l = getlinepts(lastkey, key)
                x = np.array([v[0] for v in pt_l])
                y = np.array([v[1] for v in pt_l])
                #for iter in range(x.shape[0]):
                #    polypts.append(tuple((x[iter],y[iter])))
                ed[x,y] = 1
                edcc = ed
                edcc = edcc*255
                edcc = colorize_mask(edcc)
                edcc.save("edge_c.png")
                lastkey = key
                fp.append(key)
                cnt = 0
            if (not allpts or dg) :
                flag = False
        ####### fill the poly############
        mask = getmask(polypts, mp)
        ed[mask] = 1
        edcc = ed
        edcc = edcc*255
        edcc = colorize_mask(edcc)
        edcc.save("edge_f.png")
        
        if (not allpts) :
            flag2 = False
    return ed

def poly(mask) :
    m2 = mask
    m2 = m2 * 255
    m2 = m2.astype(np.uint8)
    m2 = colorize_mask(m2)
    m2.save("cid.png")
    
    ###obtain boundaries######
    mask_pad = np.pad(mask, ((1, 1), (1, 1)), mode='constant', constant_values=0)
    edge = np.zeros(mask.shape)

    dist = distance_transform_edt(mask_pad) ## distance of each foreground pixel to a background pixel, 1 dis -- border
    dist = dist[1:-1, 1:-1]
    dist[dist > 1] = 0

    edge = (dist > 0).astype(np.uint8)
    edget = edge
    edge = edge * 255
    edge = colorize_mask(edge)
    edge.save("edge.png")
    
    loc = np.argwhere(edget > 0)  #### take a starting point, follow a path, take distant points, join the line 
    ma = getpoly(loc, edget)
    return ma 
    
f1 = open('trainade_1.txt', 'r')
add = "data/ADE20k/ADEChallengeData2016/annotations/training/"

Lines = f1.readlines()
cnt = 0

iou_acc = 0
mac = 0
### get instance information too
a = int(sys.argv[1])
#b = int(sys.argv[2])

### individual erosion for individual classes ###
for line in Lines[15*(a-1):15*a]:
#for line in Lines:
    fn = line.strip()

    mask_path = add + fn + ".png"

    cnt += 1
    print("doing for ", cnt, line)
    mask_fine = Image.open(mask_path)

    mask_check = colorize_mask(np.array(mask_fine))
    mask_check.save("color_mask.png")
    mask_fine = np.array(mask_fine) 
    mask_lb = mask_fine

    if os.path.exists("ADE_coarse/" + fn + ".png"):
        continue

    test = 255 * np.ones(mask_lb.shape)
    
    ## 0 is ignore class in ADE20k
    for i in range(1,151) :
        test2 = np.zeros(mask_lb.shape)
        mask1 = mask_lb == i
        if (np.sum(mask1) > 0) :
            test2[mask1] = 1
            ## here we should get the connected components
            ## and perform eros/dilation individually on 
            ## the connected components 
            labeled, num_features = label(test2)

            for iter in range(num_features):
                test3 = (labeled == iter + 1).astype(np.uint8)
                if (test3 > 0).sum() < 100: ## do not do any labelling for masks < 100 pixels
                    continue
                test3 = performeros(test3)
                test3 = poly(test3)
                test[test3 == 1] = i
    
    test2 = colorize_mask(test)
    test = Image.fromarray(test.astype(np.uint8))
    test.save("ADE_coarse/" + fn + ".png")
    test2.save("ADE_coarse_color/" + fn + ".png")
    #sys.exit()
    
