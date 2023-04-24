from config import *
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image,ImageDraw, ImageFont
import math
from yolov5 import train, detect
from yolov5.detect import non_max_suppression
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.torch_utils import select_device
from roi_selection import selectBrownScoreBasedROIs
import torch
from tqdm import tqdm
from yolov5.utils.augmentations import (Albumentations, augment_hsv, classify_albumentations, classify_transforms, copy_paste,
                                 letterbox, mixup, random_perspective)


'''
used to visualize selected weakboxes for each path
input:
susbseth5file- where the h5 for weaklabels are present
weakpatchoutputdir- where to store patch images
both given in config.py
output:
write the patch(for each patch used in exp) with weak bboxes and label count to weakpatchoutputdir
'''
def visualizeWeakbboxes():
    #read already create weaklabels
    newds=h5py.File(susbseth5file, 'r')
    if not os.path.exists(weakpatchoutputdir):
        os.makedirs(weakpatchoutputdir)
    font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 28, encoding="unic")

    for i,imgname in enumerate(newds['indROIs'].keys()):
        im_arr=newds['x'][i]
        im=Image.fromarray(im_arr)
        drawim=ImageDraw.Draw(im)
        for bbox in newds['indROIs'][imgname]['bboxes'][:]:
            x1, y1, x2, y2 = bbox
            #print(bbox)
            drawim.rectangle([(x1,y1),(x2,y2)],outline=(255,0,10),width=3)
            drawim.text((1,1,1,1),str(newds['y'][i]),fill=(255,0,109),font=font)
        im.save(os.path.join(weakpatchoutputdir,imgname+'.jpg'))
    newds.close()
        
def visualizeIndividualClusterinDir(newds,labels,alsoYolo):
    for cluster_n in range(n_clusters):
        if not os.path.exists(os.path.join(visualizationdir,'cluster_'+str(cluster_n))):
            os.makedirs(os.path.join(visualizationdir,'cluster_'+str(cluster_n)))
    lcount=0
    for i,imgname in enumerate(newds['indROIs'].keys()):
        drawim=newds['x'][i]
        drawim=Image.fromarray(drawim)
        for bbox in newds['indROIs'][imgname]['bboxes'][:]:
            if lcount >= len(labels):
                continue
            x1, y1, x2, y2 = bbox
            cropim=drawim.crop((x1, y1, x2, y2))
            cropim.save(os.path.join(visualizationdir,'cluster_'+str(labels[lcount]),imgname+'_'+str(lcount)+'.png'))
            lcount+=1
        if(alsoYolo):
            if not os.path.exists(os.path.join(visualizationdir,'yolo_cluster_'+str(cluster_n))):
                os.makedirs(os.path.join(visualizationdir,'yolo_cluster_'+str(cluster_n)))
            #check if imgname is in yoloROIs
            if imgname not in newds['yoloROIs'].keys():
                continue


            for bbox in newds['yoloROIs'][imgname]['bboxes'][:]:
                if lcount >= len(labels):
                    continue
                x1, y1, x2, y2 = bbox
                cropim=drawim.crop((x1, y1, x2, y2))
                cropim.save(os.path.join(visualizationdir,'yolo_cluster_'+str(labels[lcount]),imgname+'_'+str(lcount)+'.png'))
                lcount+=1
    
    for cluster_n in range(n_clusters):
        visualizeclusterimgs(os.path.join(visualizationdir,'cluster_'+str(cluster_n)))

   

def visualizeclusterimgs(clusterImgDir):
    num_imgs=len(os.listdir(clusterImgDir))
    if(num_imgs>4):
        cols=4
        rows=math.ceil(num_imgs/4)
    else:
        rows=2
        cols=num_imgs

    
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

def yolo_to_x1y1x2y2(yolo_bbox):
    x_center, y_center, width, height = yolo_bbox[0],yolo_bbox[1],yolo_bbox[2],yolo_bbox[3]
    x1 = x_center - (width / 2)
    y1 = y_center - (height / 2)
    x2 = x_center + (width / 2)
    y2 = y_center + (height / 2)
    return x1, y1, x2, y2


def convert_yolo_bboxes(bounding_boxes,img_width,img_height):
    yolo_boxes = []
    for box in bounding_boxes:
        x1, y1, x2, y2 = box[0],box[1],box[2],box[3]
        w = x2 - x1
        h = y2 - y1
        center_x = float(x1 + w/2) / img_width
        center_y = float(y1 + h/2) / img_height
        width = float(w) / img_width
        height = float(h) / img_height
        yolo_boxes.append([center_x, center_y, width, height])
    
    return yolo_boxes

'''
def yoloLabeling(itrno:int,numofimages):
    newds=h5py.File(susbseth5file, 'r+')
    #check if its already there so to delete
    try:
        
        del newds['yoloROIs']
        
        yoloROIs=newds.create_group('yoloROIs')

    except ValueError as e:
        print('error',e)
        yoloROIs=newds.create_group('yoloROIs')
    except KeyError as e:
        print('error',e)
        yoloROIs=newds.create_group('yoloROIs')
    
    device=select_device()
    weights=os.path.join(yolotraining_dir,'itr_'+str(itrno-1),'exp','weights','best.pt')
    data=os.path.join(yolodir,'training','dataset.yaml')
    model = DetectMultiBackend(weights, device=device, dnn=False, fp16=False)
    
    ds=h5py.File(lystoh5file, 'r')
    selected_imgs = ds['x'][:numofimages,16:-16,16:-16,:]
    selected_labels=ds['y'][:numofimages]
    print('Infereing for clustering later')
    for i, img in enumerate(tqdm(selected_imgs)):
        bounding_boxes_list=[]
        feature_for_bboxes=[]
        
        img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img1 = cv2.resize(img1, (416, 416))
        img1 = img1.astype(np.float32)
        img1 = img1 / 255.0
        img1 = np.expand_dims(img1, axis=0)
        img1 = torch.from_numpy(img1)
        img1 = img1.permute(0, 3, 1, 2)
        img1 = img1.to(device)
        
        
        bounding_boxes = model(img1)
        
        bounding_boxes=non_max_suppression(bounding_boxes, 0.4, 0.45, None, False, max_det=1000)
        bounding_boxes = bounding_boxes[0].tolist()

        if len(bounding_boxes)<=0:
            continue
        
        
        for roi in bounding_boxes:
             roi = yolo_to_x1y1x2y2(roi)
             bounding_boxes_list.append(roi)
             im=Image.fromarray(img)
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
        if len(feature_for_bboxes)<=0:
            continue
        forim=yoloROIs.create_group('img_'+str(i))
        forim.create_dataset('bboxes',data=bounding_boxes_list)
        forim.create_dataset('features',data=feature_for_bboxes)
        #print(forim)
        #print(newds['yoloROIs']['img_'+str(i)]['bboxes'])
    newds.close()

'''


def yoloLabeling(itrno:int,numofimages):
    newds=h5py.File(susbseth5file, 'r+')
    #check if its already there so to delete
    try:
        yoloROIs=newds['yoloROIs']
        del newds['yoloROIs']
        yoloROIs=newds.create_group('yoloROIs')

    except KeyError as e:
        yoloROIs=newds.create_group('yoloROIs')
    
    device=select_device()
    weights=os.path.join(yolodir,'training','itr_'+str(0),'exp','weights','best.pt')
    data=os.path.join(yolodir,'training','dataset.yaml')
    model = DetectMultiBackend(weights, device=device, dnn=False, data=data, fp16=False)
    # model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    ds=h5py.File(lystoh5file, 'r')
    i=0
    selected_imgs = ds['x'][:numofimages,16:-16,16:-16,:]
    selected_labels=ds['y'][:numofimages]

    #print(selected_labels[0])

    first = True
    # model.warmup(imgsz=(1 if model.pt or model.triton else 1, 3, *imgsz))
    for i, img in enumerate(selected_imgs):
        forim=yoloROIs.create_group('img_'+str(i))
        # im=Image.fromarray(img)
        im = letterbox(img, (288,288), stride=32, auto=True)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)

        im = torch.from_numpy(im).to(model.device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        pred = model(im)
        conf_thres=0.25  # confidence threshold
        iou_thres=0.45  # NMS IOU threshold
        max_det=1000
        agnostic_nms=False
        classes=None
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        bounding_boxes = []
        for i, det in enumerate(pred):

            gn = torch.tensor(img.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], img.shape).round()
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    # s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                save_conf = False
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    # with open(f'{txt_path}.txt', 'a') as f:
                    #     f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    # print(('%g ' * len(line)).rstrip() % line )
                    bounding_boxes.append(list(line[1:]))
            # first = False
        # print(bounding_boxes)
        forim.create_dataset('bboxes',data=bounding_boxes)
        print(forim)
        print(newds['yoloROIs']['img_'+str(i)]['bboxes'])
    newds.close()
    
    

'''
    given a set of images it identifies bboxes based on features of stains of images

    input:
    selected_imgs- np nest arrays of list of images
    selected_labels- np array of labels (both follow lysto fomat specified in
    https://lysto.grand-challenge.org/
    susbseth5file- path to store the h5file. given in config.py
    )
    output:
    susbseth5file- a h5 file in the given path
'''
def weakLabeling(selected_imgs,selected_labels):
    newds=h5py.File(susbseth5file, 'w')
    subdataset=newds.create_dataset('x',data=selected_imgs)
    subdataset1=newds.create_dataset('y',data=selected_labels)
    indWeakROIs=newds.create_group('indROIs')

    #groups for individual bboxes

    for i, img in enumerate(tqdm(selected_imgs)):
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

        #bboxes_from_yolo=fromyolo(img)
        

        #print('For ',str(selected_imgs),'images there are ',len(selected_weak_bboxes),'weak bboxes')
        forim.create_dataset('bboxes',data=selected_weak_bboxes)
        forim.create_dataset('features',data=feature_for_bboxes)
    newds.close()

def trainyolo(iter_round):
    yoloallimages=os.path.join(yoloalldata,'images')
    yoloalltxt=os.path.join(yoloalldata,'txt')
    yolotraining_data_dir=os.path.join(yolotraining_dir,'data')
    #remove the splits as we will copy it separately
    if(os.path.exists(yolotraining_data_dir)):
        shutil.rmtree(yolotraining_data_dir)
        

    training_images_path = os.path.join(yolotraining_data_dir, 'training','images')
    validation_images_path =os.path.join(yolotraining_data_dir, 'validation','images')
    training_labels_path = os.path.join(yolotraining_data_dir, 'training','labels')
    validation_labels_path =os.path.join(yolotraining_data_dir, 'validation','labels')
    
    os.makedirs(training_images_path)
    os.makedirs(validation_images_path)
    os.makedirs(training_labels_path)
    os.makedirs(validation_labels_path)

    files = []

    ext_len = len(extension_allowed)

    for r, d, f in os.walk(yoloallimages):
        for file in f:
            if file.endswith(extension_allowed):
                strip = file[0:len(file) - ext_len]      
                files.append(strip)

    size = len(files)
    split = int(split_percentage * size / 100)
    print("copying training data")
    for i in range(split):
        strip = files[i]
                            
        image_file = strip + extension_allowed
        src_image = os.path.join(yoloallimages,image_file)

        shutil.copy(src_image, training_images_path) 
                            
        annotation_file = strip + '.txt'
        src_label = os.path.join(yoloalltxt,annotation_file)
        shutil.copy(src_label, training_labels_path) 

    print("copying validation data")
    for i in range(split, size):
        strip = files[i]
                            
        image_file = strip + extension_allowed
        src_image = os.path.join(yoloallimages,image_file)
        shutil.copy(src_image, validation_images_path) 
                            
        annotation_file = strip + '.txt'
        src_label = os.path.join(yoloalltxt,annotation_file)
        shutil.copy(src_label, validation_labels_path) 

    print("finished copying")
    
    f = open(os.path.join(yolotraining_dir,"dataset.yaml"), "w")

    f.write("train: "+os.path.abspath(training_images_path)+"\n")
    f.write("val: "+os.path.abspath(validation_images_path)+"\n")
    f.write("nc: 1\n")
    f.write("names: ['lymphocyte']\n")
    #f.write('save_dir:'+os.path.abspath(os.path.join(yolotraining_dir,"itr_"+str(iter_round))))
    f.close()

    print('training yolo')
    
    train.run(project=os.path.abspath(os.path.join(yolotraining_dir,"itr_"+str(iter_round))),data=os.path.join(yolotraining_dir,"dataset.yaml"),imgsz=267,epochs=50,batch_size=32,weights="yolov5s.pt")

    

    
def saveimgsforYolo(selected_cluster_n,labels):
    
    if(os.path.isdir(yoloalldata)):
        shutil.rmtree(yoloalldata)
    yoloallimages=os.path.join(yoloalldata,'images')
    yoloalltxt=os.path.join(yoloalldata,'txt')
    newds=h5py.File(susbseth5file, 'r')
    
    
    os.makedirs(yoloallimages)
    
    os.makedirs(yoloalltxt)
    
    lcount=0
    for i,imgname in enumerate(newds['indROIs'].keys()):
        #print(len(newds['x']))
        drawim=newds['x'][i]
        drawim=Image.fromarray(drawim)
        imgSaved=False
        bounding_boxes=[]
        for bbox in newds['indROIs'][imgname]['bboxes'][:]:
            if lcount >= len(labels):
                continue
            if(labels[lcount]==selected_cluster_n):
                if(not imgSaved):
                    imgSaved=True
                    #saveimage
                    drawim.save(os.path.join(yoloallimages,imgname+'.jpg'))
                x1, y1, x2, y2 = bbox
                w = x2 - x1
                h = y2 - y1
                center_x = float(x1 + w/2) / drawim.width
                center_y = float(y1 + h/2) / drawim.height
                width = float(w) / drawim.width
                height = float(h) / drawim.height
                bounding_boxes.append([center_x, center_y, width, height])
            lcount+=1
        if(imgSaved):
            # Define the path to the text file that will contain the YOLO format annotations
            txt_file = os.path.join(yoloalltxt, imgname+'.txt')
             # Open the text file for writing
            with open(txt_file, "w") as f:
        # Loop over the bounding box coordinates
                for bbox in bounding_boxes:
            # Convert the bounding box coordinates to YOLO format
                    x_center, y_center, w, h = bbox
                    x_yolo = x_center
                    y_yolo = y_center
                    w_yolo = w
                    h_yolo = h
            
            # Write the YOLO format annotation to the text file
                    f.write(f"0 {x_yolo} {y_yolo} {w_yolo} {h_yolo}\n")
    newds.close()
    
   
def yolodetection(iter_round):
    yoloallimages=os.path.join(yoloalldata,'images')
    #remove the dir if it there
    yoloprediction_dir=os.path.join(yolodir,'prediction')
    if(os.path.exists(yoloprediction_dir)):
        shutil.rmtree(yoloprediction_dir)
    os.makedirs(yoloprediction_dir)
    weights=os.path.join(yolotraining_dir,'itr_'+str(iter_round),'exp','weights','best.pt')
   
    print('detecting yolo')
    
    detect.run(source = yoloallimages,project=os.path.abspath(os.path.join(yoloprediction_dir,"itr_"+str(iter_round))),data=os.path.join(yolotraining_dir,"dataset.yaml"),imgsz=(267,267),weights=weights)
