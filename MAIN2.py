
import numpy as np
import matplotlib.pyplot as plt

from skimage import data
from skimage.color import rgb2hed, hed2rgb

import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from PIL import Image
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle
import seaborn as sns
import albumentations as A
from skimage.exposure import rescale_intensity

from skimage import data
from skimage.color import rgb2hed, hed2rgb

from brown_score import *
from image_preprocessing import *
from roi_selection import *

import os

# Create output directory if it doesn't exist
if not os.path.exists('test-output'):
    os.makedirs('test-output')

# Create bbox_dir if it doesn't exist
bbox_dir = 'test-output/output'
if not os.path.exists(bbox_dir):
    os.makedirs(bbox_dir)

# Create text_dir if it doesn't exist
text_dir = 'test-output/txt'
if not os.path.exists(text_dir):
    os.makedirs(text_dir)

threshold_brown = 0.09 

import h5py

# Define the path to the input h5 file
lystoh5file='input_dir/training.h5'


# Define the object class label
object_class = "0"

# Get the number of images
numofimages = 10


import os

# Define the output directory path
output_dir = "INPUT_IMAGES"

# Create the output directory if it does not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# Specify the input and output directories
input_dir = 'input_dir'
output_dir = 'INPUT_IMAGES'

# Specify the path to the input .h5 file
input_h5_path = os.path.join(input_dir, 'training.h5')

# Open the .h5 file and get the first 10 images
with h5py.File(input_h5_path, 'r') as ds:
    selected_images = ds['x'][:numofimages,:,:,:]

# Loop over the selected images
for i, img in enumerate(selected_images):
    # Crop the image
    img = img[16:-16, 16:-16,:]
    
    # Convert the numpy array to a PIL Image object
    pil_img = Image.fromarray(img)

    
    # Save the cropped image to the output directory
    output_path = os.path.join(output_dir, f"image_{i}.png")
    pil_img.save(output_path)


input_dir = "INPUT_IMAGES"

for filename in os.listdir(input_dir):
    if not filename.endswith(".png"):
        continue
    
    # Read in image from path
    path = os.path.join(input_dir, filename)

    filename_prefix = os.path.splitext(filename)[0]

    ihc_rgb = io.imread(path)
    im = Image.open(path)

    ihc_hed = rgb2hed(ihc_rgb)

    # Create an RGB image for the DAB stain
    null = np.zeros_like(ihc_hed[:, :, 0])
    ihc_d = hed2rgb(np.stack((null, null, ihc_hed[:, :, 2]), axis=-1))

    img_height, img_width, channels = ihc_d.shape
    

    pre_processed = imagePreProcessing(path)
    processed2 = imageProcessing2(pre_processed)
    result, bounding_boxes1 = LF2(processed2,pre_processed)
    bounding_boxes_b = bounding_boxes1

    brown_scores = get_brown_scores(ihc_d, bounding_boxes_b)
    
    bounding_boxes = []
    for j in range(len(bounding_boxes_b)):
        
        if brown_scores[j] > threshold_brown:
            bounding_boxes.append(bounding_boxes_b[j])

    # Iterate through the bounding boxes
    for j, box in enumerate(bounding_boxes):
        x1, y1, x2, y2 = box
        # Crop the image to the bounding box
        cropped_im = im.crop((x1, y1, x2, y2))
        ## Resize the cropped image to 28x28
        resized_im = cropped_im.resize((28, 28))
        # Save the cropped image with a unique name
        resized_im.save(f"{bbox_dir}/image-{filename_prefix}_box_{j}.png")        



    # Convert bounding box coordinates to YOLO format
    yolo_boxes = []
    for box in bounding_boxes:
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1
        center_x = float(x1 + w/2) / img_width
        center_y = float(y1 + h/2) / img_height
        width = float(w) / img_width
        height = float(h) / img_height
        yolo_boxes.append([center_x, center_y, width, height])

    #print(bounding_boxes)
    #print(f"image_{i}.png,",bounding_boxes)


    # Define the path to the text file that will contain the YOLO format annotations
    txt_file = os.path.join(text_dir, f"{filename_prefix}.txt")


    # Open the text file for writing
    with open(txt_file, "w") as f:
        # Loop over the bounding box coordinates
        for bbox in yolo_boxes:
            # Convert the bounding box coordinates to YOLO format
            x_center, y_center, w, h = bbox
            x_yolo = x_center
            y_yolo = y_center
            w_yolo = w
            h_yolo = h
            
            # Write the YOLO format annotation to the text file
            f.write(f"{object_class} {x_yolo} {y_yolo} {w_yolo} {h_yolo}\n")

####CLUSTERING######

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
import keras.utils as image
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
import shutil

# Load the ResNet50 model
model = ResNet50(weights='imagenet', include_top=False)

# Specify the path of the folder containing the images
folder_path = 'test-output/output'

# Initialize an empty list to store the feature vectors
feature_vectors = []

# Iterate through all the images in the folder
for filename in os.listdir(folder_path):
    # If the file is an image file
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Load the image
        img = image.load_img(os.path.join(folder_path, filename), target_size=(50, 50))
        # Preprocess the image for the ResNet50 model
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        # Extract the feature vector from the image using the ResNet50 model
        feature_vector = model.predict(x).flatten()
        # Append the feature vector to the list
        feature_vectors.append(feature_vector)

# Convert the list of feature vectors to a numpy array
feature_vectors = np.array(feature_vectors)

# Normalize the feature vectors using StandardScaler
scaler = StandardScaler()
feature_vectors = scaler.fit_transform(feature_vectors)

# Perform K-means clustering on the feature vectors
kmeans = KMeans(n_clusters=2, random_state=0).fit(feature_vectors)

# Create separate folders for each cluster
cluster1_path = 'cluster1'
cluster2_path = 'cluster2'
os.makedirs(cluster1_path)
os.makedirs(cluster2_path)

# Iterate through all the images in the folder again
for filename in os.listdir(folder_path):
    # If the file is an image file
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Load the image
        img = image.load_img(os.path.join(folder_path, filename), target_size=(50, 50))
        # Preprocess the image for the ResNet50 model
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        # Extract the feature vector from the image using the ResNet50 model
        feature_vector = model.predict(x).flatten()
        # Normalize the feature vector using StandardScaler
        feature_vector_norm = scaler.transform(feature_vector.reshape(1, -1))
        # Predict the cluster label for the image
        label = kmeans.predict(feature_vector_norm)[0]
        # Move the image to the corresponding cluster folder
        if label == 0:
            shutil.copy(os.path.join(folder_path, filename), cluster1_path)
        elif label == 1:
            shutil.copy(os.path.join(folder_path, filename), cluster2_path)


