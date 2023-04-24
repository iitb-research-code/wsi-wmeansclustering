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
from wc_utils import trainyolo, yolodetection,yoloLabeling
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

for i in range(0,num_of_epochs):
    #if weaklabeling not done, please do
    #other way of saying, do weak labeling for first iteration
    
    
    #if(os.path.isfile(susbseth5file)==False):
    if(i==0):
        print('Weak Labeling and its feature extraction in Progress:')
        weakLabeling(selected_imgs,selected_labels)
    else:
        print('Yolo labeling and its feature extraction in Progress:')
        yoloLabeling(i,i*numofimages)


   
    
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
    if(i>0):
        for roi in newds['yoloROIs'].keys():
            for j in newds['yoloROIs'][roi]['bboxes'][:]:
                all_weak_bboxes.append(j)

        for roi in newds['yoloROIs'].keys():
            for j in newds['yoloROIs'][roi]['features'][:]:
                all_weak_features.append(j)


    # Convert the list of feature vectors to a numpy array
    all_weak_features = np.array(all_weak_features)
    
    print('Clustering %d ROIs into %d clusters'%(len(all_weak_features),n_clusters))

    # Perform K-means clustering on the feature vectors
    labels = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(all_weak_features)

    #save clusters for checking- visual
    if(i>0):
        alsoYolo = True
    else:
        alsoYolo = False
    visualizeIndividualClusterinDir(newds,labels,alsoYolo)
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
    yolodetection(i)
    

