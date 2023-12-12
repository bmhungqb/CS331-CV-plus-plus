# -*- coding: utf-8 -*-
"""
@author: bmhungqb
Using YOLOv5 we extract overlapped objects and exclude those objects.
# Using YOLOv5 to extract bounding boxes recognized by DETR as the same objects within actual data.
"""

# 1. Import Module
import os
import numpy as np
import torch
import cv2
import pandas as pd
import time
from tqdm import tqdm
import yolov5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. Setting up Data and Variables
df = pd.read_csv('preprocess-distance-data/datasets/annotations_process.csv')
train_image_list = os.listdir('preprocess-distance-data/datasets/data/image/train')

# Align labels to match the COCO dataset
df.filename = [f.replace('txt', 'png') for f in df.filename] # .txt -> .png

# 3. Model
# YOLOv5
model = yolov5.load('weights/yolov5x.pt')
model.eval()
model.to(device)
CLASSES = model.names
# 4. Algorithm (Make data)
# Using the sum of squared errors (SSE) for coordinate values
'''
there is too many object when we extract bounding box, there is too many overlapped object.
So if there is another exception, we calculate IOU and extract it.

Algorithm Objective: The algorithm aims to address the situation where yolov5 predicts more bounding boxes than those present in the actual data.
 The goal is to compare yolov5's predicted boxes to identify which ones match the real data. 
 This comparison facilitates the utilization of the data's zloc (presumably a localization parameter) in the preprocessing stage.

Method: The algorithm employs a method based on comparing the sum of squared differences in coordinates between bounding boxes.
 If the values are closest, it is considered that yolov5 recognizes these bounding boxes as representing the same object.
  In cases where multiple bounding boxes from yolov5 are deemed to represent a single object, indicating duplication,
   the associated data is excluded and not used.
'''

start = time.time() # Start time measurement

# Final DataFrame
glp_kitti_preprocessing_data = pd.DataFrame()

# Desired image
for k in tqdm(range(len(train_image_list))):
    # Check progress
    print('Processing {} out of {} images in total.'.format(len(train_image_list), k+1))

    mask = df['filename'] == train_image_list[k]
    df_choose = df.loc[mask]
    #print(df_choose)
    
    # Class and coordinate values of Real data
    class_list = df_choose[['class']].values
    coordinates = df_choose[['xmin', 'ymin', 'xmax', 'ymax']].values

    # Open the image and create a Variable
    img_path = os.path.join('preprocess-distance-data/datasets/data/image/train/', train_image_list[k])
    img = cv2.imread(img_path)
    img_shape = img.shape

    # Prediction
    results = model(img)
    predictions = results.pred[0]
    boxes = predictions[:, :4]
    scores = predictions[:, 4]
    categories = predictions[:, 5]
    boxes = boxes.cpu().detach().numpy()  # Switch to CPU

    input_coordinates = []  # yolov5's bounding box
    label = []  # yolo label

    count = boxes.shape[0]
    if count == 0:
        continue
    else:
        # Method for calculating BBOX intersection (Distance calculation)
        for (real_xmin, real_ymin, real_xmax, real_ymax) in coordinates.tolist():
            real_coord_array = np.repeat(np.array((real_xmin, real_ymin, real_xmax, real_ymax)).reshape(1, 4), count, axis=0)
            # Subtracting each coordinate and finding the smallest value
            result = np.sum(np.square(boxes - real_coord_array), axis=1)
            index = result.argmin()
            input_coordinates.append(boxes[index])
            label.append(CLASSES[int(categories[index])])

    input_coordinates = np.array(input_coordinates)
    
    # Create an arbitrary DataFrame
    glp_df = pd.DataFrame({
        'filename': df_choose['filename'],
        'class': label,
        'real_class': df_choose['class'],
        'xmin': input_coordinates[:, 0],
        'ymin': input_coordinates[:, 1],
        'xmax': input_coordinates[:, 2],
        'ymax': input_coordinates[:, 3],
        'angle': df_choose['observation angle'],
        'zloc': df_choose['distance']
    })
    # Remove duplicate data
    glp_df.drop_duplicates(['xmin', 'ymin', 'xmax', 'ymax'], inplace=True)  # keep=False

    glp_df = glp_df.loc[glp_df['class'] == glp_df['real_class']]  # Exclude if class is different
    # glp_df.reset_index(inplace=True)
    # glp_df.drop('index', inplace=True, axis=1)

    # Merge data
    glp_kitti_preprocessing_data = pd.concat([glp_kitti_preprocessing_data, glp_df], axis=0)

print('Finish')
end = time.time()  # End time measurement
print(f"{end - start:.5f} sec")

# Information
print(glp_kitti_preprocessing_data.head(10))
print(glp_kitti_preprocessing_data.tail(10))
print(glp_kitti_preprocessing_data.info())
glp_kitti_preprocessing_data.isnull().sum(axis=0)  # Check for NA values

# Final save
# glp_kitti_preprocessing_data = pd.read_csv('./datasets/glp_kitti_preprocessing_data.csv')
glp_kitti_preprocessing_data.isnull().sum(axis=0)
glp_kitti_preprocessing_data.drop('real_class', axis=1, inplace=True)
glp_kitti_preprocessing_data['weather'] = 'clone'

glp_kitti_preprocessing_data.to_csv('./datasets/detr_kitti_preprocessing_data_iou_remove2.csv')  # Save as CSV
