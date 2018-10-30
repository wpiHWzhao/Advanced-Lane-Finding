## Udacity Project: Advance Lane Finding.
# The lane memory to smoothen the result.
# Developed by Haowei Zhao, Oct, 2018.


import numpy as np
from collections import deque

def creat_lane_list():
    return deque(maxlen=15)

def creat_lane_list_real():
    return deque(maxlen=30)

def left_lane_add(left_lane_que,left_fit,left_bottom):
    left_lane_que.append([left_fit[0],left_fit[1],left_fit[2],left_bottom])
    return left_lane_que

def right_lane_add(right_lane_que,right_fit,right_bottom):
    right_lane_que.append([right_fit[0],right_fit[1],right_fit[2],right_bottom])
    return right_lane_que

def left_lane_mean(left_lane_que):
    if len(left_lane_que) == 0:
        return [0,0,0],0

    left_lane_mean_para = np.mean(left_lane_que,axis=0)
    left_fit_mean = [left_lane_mean_para[0],left_lane_mean_para[1],left_lane_mean_para[2]]
    left_bottom_mean = left_lane_mean_para[3]

    return left_fit_mean, left_bottom_mean


def right_lane_mean(right_lane_que):
    if len(right_lane_que) == 0:
        return [0, 0, 0], 0

    right_lane_mean_para = np.mean(right_lane_que, axis=0)
    right_fit_mean = [right_lane_mean_para[0], right_lane_mean_para[1], right_lane_mean_para[2]]
    right_bottom_mean = right_lane_mean_para[3]

    return right_fit_mean, right_bottom_mean

