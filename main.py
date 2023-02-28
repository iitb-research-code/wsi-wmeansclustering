import h5py
import os
from roi_selection import selectBrownScoreBasedROIs
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
img = ds['x']



#get the number of images
selected_images = img[:numofimages,:,:,:]

selected_images_bboxes=[]

for img in selected_images:
    #weak supervision roiselection
    bboxes=selectBrownScoreBasedROIs(img,brown_score_threshold)
    selected_images_bboxes.append(bboxes)

newds = h5py.File(susbseth5file, 'w')  
newds.create_dataset('x',selected_images, compression="gzip")
newds.create_dataset('bboxes',selected_images_bboxes, compression="gzip")
newds.close()


#clustering
#clustering(selected_images_bboxes)