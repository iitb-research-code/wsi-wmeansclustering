import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image,ImageDraw
import math

def visualizeWeakbboxes(h5pyfile,outputdir):
    for i,imgname in enumerate(h5pyfile['indROIs'].keys()):
        im_arr=h5pyfile['x'][i]
        im=Image.fromarray(im_arr)
        drawim=ImageDraw.Draw(im)
        for bbox in h5pyfile['indROIs'][imgname]['bboxes'][:]:
            x1, y1, x2, y2 = bbox
            #print(bbox)
            drawim.rectangle([(x1,y1),(x2,y2)],outline=(255,0,10),width=3)
        im.save(os.path.join(outputdir,imgname+'.jpg'))
        


def savclusterimg(clusterImgDir):
    num_imgs=len(os.listdir(clusterImgDir))
    cols=4
    rows=math.ceil(num_imgs/4)
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(200,200))
    img_count = 0
    img_arr = []
    for imf in os.listdir(clusterImgDir):
        img=Image.open(os.path.join(clusterImgDir,imf))
        img=img.resize((28,28))
        img=np.array(img)
        img_arr.append(img)

    for i in range(rows):
        for j in range(cols):
            if img_count < num_imgs:
                axes[i,j].imshow(img_arr[img_count])
                img_count+=1
    imname=clusterImgDir.split('/')[-1].split('.')[0]+'.jpg'
    plt.savefig(os.path.join(clusterImgDir,'..',imname))

def convert_yolo_bboxes(bounding_boxes,img_width,img_height):
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