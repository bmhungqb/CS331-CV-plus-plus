import pandas as pd
import matplotlib.pyplot as plt
origin_anno_path = 'preprocess-distance-data/datasets/annotations_process.csv'
annos = pd.read_csv(origin_anno_path)
distances = annos['distance']
distances.hist(bins=1000)
plt.xticks(range(0, int(max(distances)), 5))  # Set the x-axis ticks to [0, 5, 10, 15, ...]
plt.show()
