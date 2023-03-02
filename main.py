import h5py
import os
from roi_selection import selectBrownScoreBasedROIs
from PIL import Image
import numpy as np
from skimage.color import rgb2hed, hed2rgb
from utils import savclusterimg
from matplotlib import pyplot as plt
import cv2
from vggnet import VGGNet
import torch
from config import *
from sklearn.cluster import KMeans
from utils import visualizeWeakbboxes

vgg_model = VGGNet(requires_grad=False, model=VGG_model)
vgg_model.eval()
use_gpu = torch.cuda.is_available()
if use_gpu:
    vgg_model = vgg_model.cuda()

def weakLabeling(selected_imgs):
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
            #if total tize is less than 512 resize it to 23*23
            if((x2-x1)*(y2-y1)<3000):
                cropim=cropim.resize((35,35))
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
                    inputs = torch.autograd.Variable(torch.from_numpy(bgrimg).float())
                d_hist = vgg_model(inputs)[pick_layer]
                d_hist = np.sum(d_hist.data.cpu().numpy(), axis=0)
                d_hist /= np.sum(d_hist)  # normalize
                feature_for_bboxes.append(d_hist)
            except Exception as e:
                print('exception in getting features',e)
                pass
        forim.create_dataset('bboxes',data=selected_weak_bboxes)
        forim.create_dataset('features',data=feature_for_bboxes)


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

#if weaklabeling not done, please do
if(os.path.isfile(susbseth5file)==False):
    weakLabeling(selected_imgs)
else:
    weakLabeling(selected_imgs)


#check if already a subset file exists
newds=h5py.File(susbseth5file, 'r')

#visualize weaklabelings
visualizeWeakbboxes(newds,'output/visual/weaklabels4patch')
exit()
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
labels = KMeans(n_clusters=2, random_state=0).fit_predict(all_weak_features)

#save clusters for checking- visual
if not os.path.exists(os.path.join(visualizationdir,'cluster_0')):
    os.makedirs(os.path.join(visualizationdir,'cluster_0'))
if not os.path.exists(os.path.join(visualizationdir,'cluster_1')):
    os.makedirs(os.path.join(visualizationdir,'cluster_1'))

lcount=0
for i,imgname in enumerate(newds['indROIs'].keys()):
    drawim=newds['x'][i]
    drawim=Image.fromarray(drawim)
    for bbox in newds['indROIs'][imgname]['bboxes'][:]:
        if lcount >= len(labels):
            continue
        x1, y1, x2, y2 = bbox
        cropim=drawim.crop((x1, y1, x2, y2))
        if(labels[lcount]==0):
            cropim.save(os.path.join(visualizationdir,'cluster_0',imgname+'_'+str(lcount)+'.png'))
        elif(labels[lcount]==1):
            cropim.save(os.path.join(visualizationdir,'cluster_1',imgname+'_'+str(lcount)+'.png'))
        lcount+=1


savclusterimg(os.path.join(visualizationdir,'cluster_0'))
savclusterimg(os.path.join(visualizationdir,'cluster_1'))
    



#clustering
#afterclustering(selected_images_bboxes)