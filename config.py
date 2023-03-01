import numpy as np

visualizationdir='output/visual'
#some global configs
#lysto h5 file
lystoh5file='input_dir/training.h5'
susbseth5file='output/subset.h5'
#number of images
numofimages=10
brown_score_threshold=0.09

VGG_model  = 'vgg19'  # model type
means = np.array([103.939, 116.779, 123.68]) / 255. # mean of three channels in the order of BGR
# configs for histogram
pick_layer = 'avg'    # extract feature of this layer

