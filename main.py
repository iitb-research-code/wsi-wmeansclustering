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
from wc_utils import trainyolo, yolodetection
import os, shutil, random
import pandas as pd


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

for ii in range(num_of_epochs):

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
    trainyolo(ii)
    

    yolodetect_dir=os.path.join('inputforyolo')
    if(os.path.exists(yolodetect_dir)):
        shutil.rmtree(yolodetect_dir)
    os.makedirs(yolodetect_dir)

    yoloresult_dir=os.path.join('resultyolo')
    if(os.path.exists(yoloresult_dir)):
        shutil.rmtree(yoloresult_dir)
    os.makedirs(yoloresult_dir)


    #infer on yolo
    img_path_train= os.path.join(yolodir,'images')
    
    validation_img = ds['x'][:numofimages,16:-16,16:-16,:]
    folder_path = yolodetect_dir
    # Iterate through the selected images
    for i, img in enumerate(validation_img):
        # Crop the image to remove the 16 pixel border
        img = img[16:-16, 16:-16]
        img = Image.fromarray(img)
        # Save the image to the folder path
        file_path = os.path.join(folder_path, "img_{}.jpg".format(i))
        img.save(file_path)



    validation_lbl = ds['y'][:numofimages]

    # Create a list of image names
    img_names = [f"img_{i}" for i in range(numofimages)]
    # Create a dataframe with two columns: Image and Label
    data = {'Image': img_names, 'Ground_Label': validation_lbl}
    df = pd.DataFrame(data)

    
    # Save the dataframe to a CSV file in a specific folder
    folder_path = yoloresult_dir
    file_path = f'{folder_path}/labels_ground_truth.csv'
    df.to_csv(file_path, index=False)

    
    img_path_val = yolodetect_dir
    #yolodetection(i,img_path_train)
    yolodetection(ii,img_path_val)


   

    # Fixed folder path
    folder_path = yoloresult_dir

    # Input file names
    file1_name = 'labels_ground_truth.csv'
    file2_name = 'label_yolo_pred.csv'

    # Output file name
    file3_name = 'PRECISION-RECALL_0.30_50epoch_4clusters.csv'

    # Read the CSV files into dataframes
    df1 = pd.read_csv(os.path.join(folder_path, file1_name), header=None)
    df2 = pd.read_csv(os.path.join(folder_path, file2_name), header=None)

    # Merge the dataframes on the first column
    merged_df = pd.merge(df1, df2, on=0)

   


    # Write the merged dataframe to a new CSV file
    merged_df.to_csv(os.path.join(folder_path, file3_name), index=False)










