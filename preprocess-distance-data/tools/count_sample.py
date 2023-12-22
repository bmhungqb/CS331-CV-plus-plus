import pandas as pd
import matplotlib.pyplot as plt
origin_anno_path = 'preprocess-distance-data/datasets/annotations_process.csv'
annos = pd.read_csv(origin_anno_path)
print(annos.count(axis=0))