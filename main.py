from Source.DTW import DTW

import argparse
import json
import numpy as np


json_test_path = '/content/gdrive/MyDrive/test_golf/skeleton_test.json'

with open(json_test_path, 'r') as f:
    a = json.load(f)

# pose1 = np.array(a['2'])
# pose2 = np.array(a['4'])

dtw = DTW()

# e = dtw.compare_1_1(pose1, pose2, 'allign')
# e = 100 - e.mean(axis = 1) / np.pi * 100                                                                 
# print('point by allignment: ', e)

# e, e_ = dtw.compare_1_1(pose1, pose2, 'rotate')
# e = 100 - e / np.pi * 100
# # e_ = 100 - e_.mean(axis = 1) / np.pi * 100
# print(f'point by rotation: {e} at {e_}')

pose1 = np.array(a['6'])
pose_rotate = dtw.euler_rotation(pose1[0],90, 'X')[np.newaxis, ...]

# e = dtw.compare_1_1(pose_rotate, pose1, 'allign')
# e = 100 - e.mean(axis = 1) / np.pi * 100                                                                 
# print('point by allignment: ', e)

# e, e_ = dtw.compare_1_1(pose_rotate, pose1, 'rotate')
# e = 100 - e / np.pi * 100
# # e_ = 100 - e_.mean(axis = 1) / np.pi * 100
# print(f'point by rotation: {e} at {e_}')

f = dtw(pose_rotate, pose1)
print(f'Test {f}') 