import numpy as np
import h5py
import os
import shutil
import torch
from vggnet import VGGNet

#some global configs

#lysto h5 file
lystoh5file=os.path.join('input_dir','training.h5')

experiment_name='test'
num_of_epochs=3

outputdir=os.path.join('output',experiment_name)

if(os.path.isdir(outputdir)):
    print('Warning: Directory with experiment name %s exists. Preferable to have an empty directory to avoid stale files')
else:
    os.makedirs(outputdir)

visualizationdir=os.path.join(outputdir,'visual')
os.makedirs(visualizationdir,exist_ok=True)
weakpatchoutputdir=os.path.join(visualizationdir,'weakpatch')

susbseth5file=os.path.join(outputdir,'subset.h5')
#number of images
numofimages=100
brown_score_threshold=0
n_clusters=2

#yolo
yolodir=os.path.join(outputdir,'yolotrain')
yolodirimages=os.path.join(yolodir,'images')
yolodirtxt=os.path.join(yolodir,'txt')
extension_allowed = '.jpg'

split_percentage = 90   

VGG_model  = 'vgg19'  # model type
means = np.array([103.939, 116.779, 123.68]) / 255. # mean of three channels in the order of BGR
# configs for histogram
pick_layer = 'avg'    # extract feature of this layer

#declare the model used for feature extraction
vgg_model = VGGNet(requires_grad=False, model=VGG_model)
vgg_model.eval()
use_gpu = torch.cuda.is_available()
if use_gpu:
    vgg_model = vgg_model.cuda()


h5py.get_config().track_order=True