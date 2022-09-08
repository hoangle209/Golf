from Source.DTW import DTW

import argparse
import json
import numpy as np


json_test_path = '/content/gdrive/MyDrive/test_golf/skeleton_test.json'
# json_test_path = '/content/gdrive/MyDrive/golf_data/skeleton_perfect.json'
with open(json_test_path, 'r') as f:
    a = json.load(f)

# pose1 = np.array(a['8'])
# pose2 = np.array(a['100'])

pose1 = np.array(a['5'])
pose2 = np.array(a['6'])

dtw = DTW()

point = dtw(pose1, pose2, allignment = 'allign')
for k, v in point.items():
  print(f'{k}: {v}')



# pose1 = np.array(a['1'])
# pose_rotate = dtw.euler_rotation(pose1[0],280, 'Z')[np.newaxis, ...]

# e = dtw.compare_1_1(pose_rotate, pose1, 'allign')
# e = 100 - e.mean(axis = 1) / np.pi * 100                                                                 
# print('point by allignment: ', e)

# e, e_ = dtw.compare_1_1(pose_rotate, pose1, 'rotate')
# e = 100 - e / np.pi * 100
# # e_ = 100 - e_.mean(axis = 1) / np.pi * 100
# print(f'point by rotation: {e} at {e_}')

# f = dtw(pose_rotate, pose1)
# print(f'Test {f}') 