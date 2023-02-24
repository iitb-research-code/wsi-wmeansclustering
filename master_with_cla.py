import argparse
import os
from PIL import Image
from skimage.color import rgb2hed, hed2rgb
from skimage import io
import numpy as np

from brown_score import *
from image_preprocessing import *
from roi_selection import *


def parse_args():
    parser = argparse.ArgumentParser(description='Process IHC images and generate YOLO format annotations.')
    parser.add_argument('input_path', type=str, nargs='?', default='test-images', help='path to directory containing input images')
    return parser.parse_args()



def main():
    args = parse_args()

    # Define the object class label
    object_class = "0"


    # Define the path to the output folder for the cropped images
    output_folder_path = os.path.join(args.input_path, 'cropped')

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder_path, exist_ok=True)

    # Define the path to the output folder for the YOLO format annotations
    annotations_folder_path = os.path.join(args.input_path, 'annotations')

    # Create the output folder if it doesn't exist
    os.makedirs(annotations_folder_path, exist_ok=True)

 

    


    # Loop over the input images
    for i, filename in enumerate(os.listdir(args.input_path)):
        if not filename.endswith('.png'):
            continue

        
        # Read in image from path
        path = os.path.join(args.input_path, filename)

        filename_prefix = os.path.splitext(filename)[0]
        
        ihc_rgb = io.imread(path)
        im = Image.open(path)    

        # Separate the stains from the IHC image
        ihc_hed = rgb2hed(ihc_rgb)

        # Create an RGB image for the DAB stain
        null = np.zeros_like(ihc_hed[:, :, 0])
        ihc_d = hed2rgb(np.stack((null, null, ihc_hed[:, :, 2]), axis=-1))

        img_height, img_width, channels = ihc_d.shape

        # Pre-process image
        pre_processed = imagePreProcessing(path)
        processed2 = imageProcessing2(pre_processed)
        result, bounding_boxes1 = LF2(processed2, pre_processed)
        bounding_boxes_b = bounding_boxes1

        # Compute brown scores
        brown_scores = get_brown_scores(ihc_d, bounding_boxes_b)

        # Select bounding boxes with brown scores above threshold
        bounding_boxes = []
        for j in range(len(bounding_boxes_b)):
            if brown_scores[j] > 0.09:
                bounding_boxes.append(bounding_boxes_b[j])

        # Iterate through the bounding boxes
        for j, box in enumerate(bounding_boxes):
            x1, y1, x2, y2 = box

            # Crop the image to the bounding box
            cropped_im = im.crop((x1, y1, x2, y2))

            # Resize the cropped image to 28x28
            resized_im = cropped_im.resize((28, 28))

            # Save the cropped image with a unique name
            
            resized_im.save(os.path.join(output_folder_path, f"{filename_prefix}_box_{j}.png"))

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

        # Define the path to the text file that will contain the YOLO format annotations
        txt_file = os.path.join(annotations_folder_path, f"{filename_prefix}.txt")

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

