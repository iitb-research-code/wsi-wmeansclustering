# wsi-wmeansclustering

mater_without_cla.py file is used to detect and extract regions of interest (ROI) from input images and create annotations for the detected ROIs in the YOLO format.

The code takes input images from the "input_dir" directory and saves the extracted ROIs in the "test-output/output" directory as cropped images. It also creates corresponding text files containing the YOLO format annotations for each image in the "test-output/txt" directory.

The steps involved in the code are:

Import necessary libraries

Set up input and output directories

Loop over input images in the input directory

Read in image from path

Separate the stains from the IHC image

Perform image preprocessing and processing to enhance image quality and reduce noise

Apply a Local Maxima Suppression (LMS) algorithm to identify potential bounding boxes around objects in the image

Calculate brown scores for each bounding box to determine if it is likely to contain an object of interest

Filter the bounding boxes based on their brown scores to obtain a refined list of bounding boxes

Iterate through the refined bounding boxes and crop the image to the bounding box

Resize the cropped image to 28x28

Save the cropped image with a unique name in the output directory

Convert bounding box coordinates to YOLO format

Write the YOLO format annotation to a text file in the output directory

To run this code, you need to create three directories: "input_dir" to store input images, "test-output/output" to store the extracted ROIs as cropped images, and "test-output/txt" to store the YOLO format annotations. Additionally, you need to import three Python scripts named "brown_score.py", "image_preprocessing.py", and "roi_selection.py" that contain custom functions used in this code.
