from config import *
import h5py
import os
from PIL import Image
import numpy as np
from skimage.color import rgb2hed, hed2rgb
from wc_utils import visualizeclusterimgs
from matplotlib import pyplot as plt
import cv2
import torch
from sklearn.cluster import KMeans
from wc_utils import visualizeWeakbboxes, weakLabeling, visualizeIndividualClusterinDir, saveimgsforYolo
 #train on yolo
from wc_utils import trainyolo
import os, shutil, random


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

for i in range(num_of_epochs):

    #if weaklabeling not done, please do
    if(os.path.isfile(susbseth5file)==False):
        weakLabeling(selected_imgs,selected_labels)
    #else:
        #yoloLabeling(selected_imgs,selected_labels)


   
    
    newds=h5py.File(susbseth5file, 'r')
    #visualize weaklabelings
    visualizeWeakbboxes()

    all_weak_bboxes=[]
    all_weak_features=[]

    for roi in newds['indROIs'].keys():
        for j in newds['indROIs'][roi]['bboxes'][:]:
            all_weak_bboxes.append(j)

    for roi in newds['indROIs'].keys():
        for j in newds['indROIs'][roi]['features'][:]:
            all_weak_features.append(j)


    # Convert the list of feature vectors to a numpy array
    all_weak_features = np.array(all_weak_features)
    
        
    # Perform K-means clustering on the feature vectors
    labels = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(all_weak_features)

    #save clusters for checking- visual
    visualizeIndividualClusterinDir(newds,labels)
    newds.close()

    #
    print('There are ',n_clusters,' clusters.')
    print('Each of them is save @,',visualizationdir)
    selected_cluster_n=int(input('Choose one that best represents the group: \n'))

    #given images as x ROIS for each x as indROIs, labels in the same
    #order save the images and bboxes for yolo training
    
    saveimgsforYolo(selected_cluster_n,labels)

    #train on yolo
    trainyolo(i)
    
    #infer on yolo

