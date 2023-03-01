import h5py
import os
from roi_selection import selectBrownScoreBasedROIs
from PIL import Image
import numpy as np
from skimage.color import rgb2hed, hed2rgb
from utils import convert_yolo_bboxes
from matplotlib import pyplot as plt
import cv2

#some global configs
#lysto h5 file
lystoh5file='input_dir/training.h5'
susbseth5file='output/subset.h5'
#number of images
numofimages=10
brown_score_threshold=0.09

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
newds=h5py.File(susbseth5file, 'w')
subdataset=newds.create_dataset('x',data=selected_imgs)
indWeakROIs=newds.create_group('indROIs')

#groups for individual bboxes

for i, img in enumerate(selected_imgs):
    forim=indWeakROIs.create_group('img_'+str(i))
    selected_weak_bboxes=selectBrownScoreBasedROIs(img,brown_score_threshold)
    forim.create_dataset('bboxes',data=selected_weak_bboxes)
    print('hi')

    



#clustering
#afterclustering(selected_images_bboxes)