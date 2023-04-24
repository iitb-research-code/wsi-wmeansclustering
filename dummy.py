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
from wc_utils import visualizeWeakbboxes, weakLabeling, visualizeIndividualClusterinDir, saveimgsforYolo, yoloLabeling
 #train on yolo
from wc_utils import trainyolo, yolodetection
import os, shutil, random

yoloLabeling(0)

# img_path_train= os.path.join(yolodir,'test')
    
#     #validation_img = ds['x'][numofimages:numofimages+500,16:-16,16:-16,:]

#    #img_path_val = 
# yolodetection(0,img_path_train)