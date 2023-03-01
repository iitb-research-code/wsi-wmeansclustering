import cv2



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