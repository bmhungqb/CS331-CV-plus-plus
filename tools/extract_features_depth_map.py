# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from scipy import stats
import torch
class ExtractFeature:
    def __init__(self):
        pass

    # Make dataset!
    def make_dataset(self, boxes, depth_map):
        self.data = pd.DataFrame(columns=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        # BBOX input
        for (xmin, ymin, xmax, ymax) in boxes.tolist():
            '''
            Determine the distance only for objects within your own lane range using xmin and xmax
            '''
            height = ymax - ymin
            width = xmax - xmin
            if int(xmin) < 0:
                xmin = 0
            if int(ymin) < 0:
                ymin = 0
            depth_mean = depth_map[int(ymin):int(ymax), int(xmin):int(xmax)].mean()
            depth_median = np.median(depth_map[int(ymin):int(ymax), int(xmin):int(xmax)])
            depth_mean_trim = stats.trim_mean(depth_map[int(ymin):int(ymax), int(xmin):int(xmax)].flatten(), 0.2)
            depth_max = depth_map[int(ymin):int(ymax), int(xmin):int(xmax)].max()
            inp = [xmin, ymin, xmax, ymax, width, height, depth_mean_trim, depth_mean, depth_max, depth_median]
            data_list = pd.DataFrame(inp).T
            self.data = pd.concat([self.data, data_list], axis=0)
            # data preprocessing
            self.data_preprocessing(depth_map)

        return self.data


    '''
    Preprocessing
    bbox comparison: If overlapping is over 70%, remove objects beyond that,
    if our image overlaps more than 70%, we remove objects that are farther away.
    If not, exclude the overlapped portion, then recalculate depth and output the values.
    if not, exclude overlapped and calculate depth again.
    '''
    def data_preprocessing(self, prediction):
        self.data.index = [i for i in range(len(self.data))]
        xmin_list = []
        ymin_list = []
        xmax_list = []
        ymax_list = []
        for k, (xmin, ymin, xmax, ymax) in zip(self.data.index, self.data[[0, 1, 2, 3]].values):
            xmin_list.insert(0, xmin)
            ymin_list.insert(0, ymin)
            xmax_list.insert(0, xmax)
            ymax_list.insert(0, ymax)

            for i in range(len(xmin_list)-1):
                y_range1 = np.arange(int(ymin_list[0]), int(ymax_list[0]+1)) # input image
                y_range2 = np.arange(int(ymin_list[i+1]), int(ymax_list[i+1]+1)) # 다른 image와 비교
                y_intersect = np.intersect1d(y_range1, y_range2)

                # print(y_intersect)

                if len(y_intersect) >= 1:
                    x_range1 = np.arange(int(xmin_list[0]), int(xmax_list[0])+1)
                    x_range2 = np.arange(int(xmin_list[i+1]), int(xmax_list[i+1]+1))
                    x_intersect = np.intersect1d(x_range1, x_range2)

                    # print(x_intersect)

                    if len(x_intersect) >= 1: # BBOXis overlapped do the statement
                        area1 = (y_range1.max() - y_range1.min())*(x_range1.max() - x_range1.min())
                        area2 = (y_range2.max() - y_range2.min())*(x_range2.max() - x_range2.min())
                        area_intersect = (y_intersect.max() - y_intersect.min())*(x_intersect.max() - x_intersect.min())

                        if area_intersect/area1 >= 0.70 or area_intersect/area2 >= 0.70: # 70% over overlapped
                            # remove far object
                            if area1 < area2:
                                try:
                                    self.data.drop(index=k, inplace=True)
                                # if it elominated before, there is left list(xmin,ymin)
                                except:
                                    pass

                            else:
                                try:
                                    self.data.drop(index=k-(i+1), inplace=True)
                                # if it elominated before, there is left list(xmin,ymin)
                                except:
                                    pass

                        # if just a little overlapped, we fix depth_min and depth_mean
                        elif  area_intersect/area1 > 0 or area_intersect/area2 > 0:
                            if area1 < area2:
                                prediction[int(y_intersect.min()):int(y_intersect.max()), int(x_intersect.min()):int(x_intersect.max())] = np.nan # masking
                                bbox = prediction[int(ymin_list[0]):int(ymax_list[0]), int(xmin_list[0]):int(xmax_list[0])]
                                depth_mean = np.nanmean(bbox)

                                if k in self.data.index:
                                    self.data.loc[k, 7] = depth_mean

                            else:
                                prediction[int(y_intersect.min()):int(y_intersect.max()), int(x_intersect.min()):int(x_intersect.max())] = np.nan # masking
                                bbox = prediction[int(ymin_list[i+1]):int(ymax_list[i+1]), int(xmin_list[i+1]):int(xmax_list[i+1])]
                                depth_mean = np.nanmean(bbox)

                                if k-(i+1) in self.data.index:
                                    self.data.loc[k-(i+1), 7] = depth_mean
            # initialize index
            self.data.reset_index(inplace=True)
            self.data.drop('index', inplace=True, axis=1)

            return self.data