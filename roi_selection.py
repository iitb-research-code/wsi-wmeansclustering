# -*- coding: utf-8 -*-
"""ROI-selection.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1kGmaFFXuDrMnL-sYTQkjqxWzzgP4u7_f
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from PIL import Image
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
import albumentations as A
from skimage.exposure import rescale_intensity


import numpy as np
import matplotlib.pyplot as plt

from skimage import data
from skimage.color import rgb2hed, hed2rgb
from image_preprocessing import imagePreProcessing,imageProcessing2

def get_brown_score(im, x1, y1, x2, y2):

    # Check data type of input array and convert to appropriate data type
    if im.dtype == np.float64:
        im = (im * 255).astype(np.uint8)

    # Convert NumPy array to PIL Image object
    im = Image.fromarray(im)

    # Crop image to bounding box
    crop_im = im.crop((x1, y1, x2, y2))
    
    # Convert image to RGB
    crop_im = crop_im.convert("RGB")
    
    # Initialize brown pixel count
    brown_pixels = 0
    
    # Iterate over all pixels in the bounding box
    for x in range(crop_im.size[0]):
        for y in range(crop_im.size[1]):
            # Get RGB values of the pixel
            r, g, b = crop_im.getpixel((x, y))
            
            # Check if pixel is brown (within a certain range of R, G, B values)
            if 128 <= r <= 255 and 0 <= g <= 128 and 0 <= b <= 128:
                brown_pixels += 1
    
    # Calculate brown score (ratio of brown pixels to total pixels)
    brown_score = brown_pixels / (crop_im.size[0] * crop_im.size[1])
    
    return brown_score

def get_brown_scores(ihc_d, bounding_boxes):
    # Initialize list to store brown scores for each bounding box
    brown_scores = []
    
    for box in bounding_boxes:
        # Get coordinates of the bounding box
        x1, y1, x2, y2 = box
        
        # Calculate brown score for the bounding box
        brown_score = get_brown_score(ihc_d, x1, y1, x2, y2)
        
        # Add brown score to list
        brown_scores.append(brown_score)
    
    return brown_scores

def selectBrownScoreBasedROIs(img,brown_score_threshold):
    """
    This function returns bboxes given an image
    input img in nparray
    output list of bboxes
    """
    #crop
    img = img[16:-16, 16:-16,:]

    img=imagePreProcessing(img)
    otsu_threshold, image_result = cv2.threshold(
         img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    thresh2 = 255-image_result

  

    output = cv2.connectedComponentsWithStats(thresh2)
    (numLabels, labels, stats, centroids) = output
    mask = np.zeros(thresh2.shape, dtype="uint8")

    # loop over the number of unique connected component labels, skipping
    # over the first label (as label zero is the background)
    for i in range(1, numLabels):
        # extract the connected component statistics for the current label
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]


        # consider the contour properties: aspect ratio, extent, or solidity
        aspect_ratio = float(w) / h
        extent = float(area) / (w * h)
        #solidity = float(area) / cv2.contourArea(cv2.findContours(labels == i, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[0][0])
    
    
        # ensure the width, height, and area are all neither too small
        # nor too big
        #keepWidth = w > 5 and w < 30
        #keepHeight = h > 5 and h < 30
        #keepArea = area > 10

        keepWidth = w > 10 and w < 88
        keepHeight = h > 10 and h < 88
        keepArea = area > 200

        # consider the aspect ratio, extent, and solidity of the contours
        keepAspectRatio = aspect_ratio > 0.2 and aspect_ratio < 1.8
        keepExtent = extent > 0.3 and extent < 0.8
        #keepSolidity = solidity > 0.7
        # ensure the connected component we are examining passes all
        # three tests
        if all((keepWidth, keepHeight, keepArea, keepAspectRatio, keepExtent)):
            # construct a mask for the current connected component and
            # then take the bitwise OR with the mask       
            componentMask = (labels == i).astype("uint8") * 1
            mask = cv2.bitwise_or(mask, componentMask)


        
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        colour = (255, 0, 0)
        thickness = 1
        i = 0
        
        bounding_boxes = []
       # Iterate through the contours and find bounding boxes
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            bounding_boxes.append([x, y, x+w, y+h])
    
    #get brownscores for bboxes
    brownscores=get_brown_scores(img,bounding_boxes)

    filtered_ounding_boxes=[]
    for j in range(len(bounding_boxes)):
        
        if brownscores[j] > brown_score_threshold:
            filtered_ounding_boxes.append(bounding_boxes[j])
        
    return filtered_ounding_boxes


def LF2(c,d): 
    
    image = d
    img = c
    # applying different thresholding 
    # techniques on the input image
    # Otsu's thresholding after Gaussian filtering
    # Apply GaussianBlur to reduce image noise if it is required

    blur = img  #cv2.GaussianBlur(img,(5,5),0)
    #image_result = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 81, 40)

    otsu_threshold, image_result = cv2.threshold(
         blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,)
    
    thresh2 = 255-image_result

    # Use morphological operations to clean up the image
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    #thresh2 = cv2.morphologyEx(thresh2, cv2.MORPH_CLOSE, kernel)

    output = cv2.connectedComponentsWithStats(thresh2)
    (numLabels, labels, stats, centroids) = output
    mask = np.zeros(thresh2.shape, dtype="uint8")

    # loop over the number of unique connected component labels, skipping
    # over the first label (as label zero is the background)
    for i in range(1, numLabels):
        # extract the connected component statistics for the current label
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]


        # consider the contour properties: aspect ratio, extent, or solidity
        aspect_ratio = float(w) / h
        extent = float(area) / (w * h)
        #solidity = float(area) / cv2.contourArea(cv2.findContours(labels == i, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[0][0])
    
    
        # ensure the width, height, and area are all neither too small
        # nor too big
        #keepWidth = w > 5 and w < 30
        #keepHeight = h > 5 and h < 30
        #keepArea = area > 10

        keepWidth = w > 10 and w < 88
        keepHeight = h > 10 and h < 88
        keepArea = area > 200

        # consider the aspect ratio, extent, and solidity of the contours
        keepAspectRatio = aspect_ratio > 0.2 and aspect_ratio < 1.8
        keepExtent = extent > 0.3 and extent < 0.8
        #keepSolidity = solidity > 0.7
        # ensure the connected component we are examining passes all
        # three tests
        if all((keepWidth, keepHeight, keepArea, keepAspectRatio, keepExtent)):
            # construct a mask for the current connected component and
            # then take the bitwise OR with the mask       
            componentMask = (labels == i).astype("uint8") * 1
            mask = cv2.bitwise_or(mask, componentMask)


        # Multiple objects
        result = image.copy()
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        colour = (255, 0, 0)
        thickness = 1
        i = 0
        
        bounding_boxes = []
       # Iterate through the contours and find bounding boxes
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            bounding_boxes.append([x, y, x+w, y+h])
            cv2.rectangle(result, (x, y), (x+w, y+h), colour, thickness)
  

    return result, bounding_boxes