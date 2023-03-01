import h5py
import os
from roi_selection import selectBrownScoreBasedROIs
from PIL import Image
import numpy as np
from skimage.color import rgb2hed, hed2rgb
from utils import convert_yolo_bboxes
from matplotlib import pyplot as plt
import cv2
from vggnet import VGGNet
import torch

#some global configs
#lysto h5 file
lystoh5file='input_dir/training.h5'
susbseth5file='output/subset.h5'
#number of images
numofimages=10
brown_score_threshold=0.09

VGG_model  = 'vgg19'  # model type
vgg_model = VGGNet(requires_grad=False, model=VGG_model)
vgg_model.eval()
use_gpu = torch.cuda.is_available()
if use_gpu:
    vgg_model = vgg_model.cuda()
means = np.array([103.939, 116.779, 123.68]) / 255. # mean of three channels in the order of BGR
# configs for histogram
pick_layer = 'avg'    # extract feature of this layer

#load lysto dataset
"""
download lyst h5 from https://zenodo.org/record/3513571#.Y_2qw9JBwd5
and add it to input_dir folder
"""

#check if file exists
if(os.path.isfile(lystoh5file)==False):
    print('H5 dont exisits. Please check')
    SystemExit(1)




ds=h5py.File(lystoh5file, 'r')
selected_imgs = ds['x'][:numofimages,16:-16,16:-16,:]

#check if already a subset file exists
if(os.path.isfile(susbseth5file)==True):
    newds=h5py.File(susbseth5file, 'r')
    all_weak_bboxes=[]
    for i in newds['indROIs'].keys():
        for j in newds['indROIs'][i]['bboxes'][:]:
            all_weak_bboxes.append(j)
    
    print(all_weak_bboxes)
    exit(0)
    

newds=h5py.File(susbseth5file, 'w')
subdataset=newds.create_dataset('x',data=selected_imgs)
indWeakROIs=newds.create_group('indROIs')

#groups for individual bboxes

for i, img in enumerate(selected_imgs):
    forim=indWeakROIs.create_group('img_'+str(i))
    selected_weak_bboxes=selectBrownScoreBasedROIs(img,brown_score_threshold)
    im=Image.fromarray(img)
    feature_for_bboxes=[]
    for roi in selected_weak_bboxes:
        x1, y1, x2, y2 = roi
        cropim=im.crop((x1, y1, x2, y2))
        cropim=np.array(cropim)
        bgrimg = cropim[:, :, ::-1]  # switch to BGR
        bgrimg = np.transpose(bgrimg, (2, 0, 1)) / 255.
        bgrimg[0] -= means[0]  # reduce B's mean
        bgrimg[1] -= means[1]  # reduce G's mean
        bgrimg[2] -= means[2]  # reduce R's mean
        bgrimg = np.expand_dims(bgrimg, axis=0)
        try:
            if use_gpu:
                inputs = torch.autograd.Variable(torch.from_numpy(bgrimg).cuda().float())
            else:
                inputs = torch.autograd.Variable(torch.from_numpy(img).float())
            d_hist = vgg_model(inputs)[pick_layer]
            d_hist = np.sum(d_hist.data.cpu().numpy(), axis=0)
            d_hist /= np.sum(d_hist)  # normalize
            feature_for_bboxes.append(d_hist)
        except:
            pass
    forim.create_dataset('bboxes',data=selected_weak_bboxes)
    forim.create_dataset('features',data=feature_for_bboxes)



    



#clustering
#afterclustering(selected_images_bboxes)