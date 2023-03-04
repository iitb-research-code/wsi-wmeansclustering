from config import *
import h5py
import os
from PIL import Image
import numpy as np
from skimage.color import rgb2hed, hed2rgb
from utils import visualizeclusterimgs
from matplotlib import pyplot as plt
import cv2
import torch
from sklearn.cluster import KMeans
from utils import visualizeWeakbboxes, weakLabeling, visualizeIndividualClusterinDir


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
selected_labels=ds['y'][:numofimages]
#if weaklabeling not done, please do
if(os.path.isfile(susbseth5file)==False):
    weakLabeling(selected_imgs,selected_labels)
else:
    weakLabeling(selected_imgs,selected_labels)


#read already create weaklabels
newds=h5py.File(susbseth5file, 'r')

#visualize weaklabelings
visualizeWeakbboxes()

all_weak_bboxes=[]
all_weak_features=[]

for i in newds['indROIs'].keys():
    for j in newds['indROIs'][i]['bboxes'][:]:
        all_weak_bboxes.append(j)

for i in newds['indROIs'].keys():
    for j in newds['indROIs'][i]['features'][:]:
        all_weak_features.append(j)


# Convert the list of feature vectors to a numpy array
all_weak_features = np.array(all_weak_features)
   
    
# Perform K-means clustering on the feature vectors
labels = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(all_weak_features)

#save clusters for checking- visual
visualizeIndividualClusterinDir(newds,labels)

#savclusterimg(os.path.join(visualizationdir,'cluster_2'))
#savclusterimg(os.path.join(visualizationdir,'cluster_3'))





#clustering
#afterclustering(selected_images_bboxes)