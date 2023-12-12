import csv
import math
import os.path
import pandas as pd

origin_anno_path = 'preprocess-distance-data/datasets/annotations.csv'
annos = pd.read_csv(origin_anno_path)
annos['dis_x_z'] = annos.apply(lambda row: math.sqrt(row['xloc']**2 + row['zloc']**2), axis=1)
annos['distance'] = annos.apply(lambda row: round(math.sqrt(row['dis_x_z']**2 + row['yloc']**2),2), axis=1)
annos['error_dis_z'] = annos.apply(lambda row: math.sqrt(row['distance'] - row['zloc']), axis=1)
annos.to_csv('preprocess-distance-data/datasets/annotations_process.csv', index=False)