from Source.DTW import DTW

import argparse
import json
import numpy as np


# json_test_path = '/content/gdrive/MyDrive/test_golf/skeleton_test.json'
json_test_path = '/content/gdrive/MyDrive/golf_data/skeleton_perfect.json'

with open(json_test_path, 'r') as f:
    a = json.load(f)

pose1 = np.array(a['8'])
pose2 = np.array(a['100'])

# pose1 = np.array(a['5'])
# pose2 = np.array(a['6'])

dtw = DTW()

# point = dtw(pose1, pose2, allignment = 'allign')
# for k, v in point.items():
#     print(f'{k}: {v}')


# pose1 = np.array(a['1'])
# pose_rotate = dtw.euler_rotation(pose1[0],280, 'X')[np.newaxis, ...]
# pose_rotate = dtw.euler_rotation(pose1[0],130, 'Y')[np.newaxis, ...]

# point, raw = dtw(pose1, pose2, allignment = 'rotate', _type = 'series')
# for k, v in raw.items():
#     print(f'{k}: {v*180/np.pi}')

point = dtw(pose1, pose2, allignment='rotate', _type = 'series')
for k, v in point.items():
    print(f'{k}: {v}')


