import numpy as np
import math
import numba
import json
from tqdm import tqdm

class DTW():
    def __init__(self, method = 'cosin'):
        assert method in ['cosin', 'L1']
        self.method = method
        self.cost_func = lambda x, y: self.cosin(x, y) \
              if self.method == 'cosin' else np.abs(x - y)

    def cosin(self, x, y):
        '''
        This funtion calculate angle between two vector
        d: [x, y, z]
        n, m: time series (or number of states)
          :param x: d x n
          :param y: d x m

          :return: cost matrix with shape n x m
        '''
        xx = (x**2).sum(axis = 0).reshape(1, -1) # shape 1 x n
        yy = (y**2).sum(axis = 0).reshape(1, -1) # shape 1 x m
        xy = (xx.T * yy)**0.5 # shape n x m 
        cos = x.T.dot(y) / xy
        assert cos.all() <= 1 and cos.all() >= -1
        return np.arccos(cos)  
    

    def dtw(self, x, y):
        '''
        This function will calculate Dynamic Time Wrapping matrix
        batch: number of vector used for comparation (17 keypoints ==> 16 vectors)
        d: [x, y, z]
        n, m: Time series
          :param x: batch x d x n
          :param y: batch x d x m

          :return matrix, batch x n x m 
        '''
        b = x.shape[0]
        n, m = x.shape[-1], y.shape[-1] 
        cummulative_matrix = np.empty(shape = (b, n, m))
        
        for k in range(b):
          _x, _y = x[k], y[k] # shape d x n and d x m
          cost = self.cost_func(_x, _y)
          # print(cost)
          for i in range(n):
            for j in range(m):
              if i == 0 and j == 0:
                cummulative_matrix[k, i, j] = cost[i, j]
              elif i == 0:
                cummulative_matrix[k, i, j] = cost[i, j] + cummulative_matrix[k, i, j-1] 
              elif j == 0:
                cummulative_matrix[k, i, j] = cost[i, j] + cummulative_matrix[k, i-1, j]
              else:
                cummulative_matrix[k, i, j] = cost[i, j] + min(cummulative_matrix[k, i-1, j-1],
                                                              cummulative_matrix[k, i-1, j],
                                                              cummulative_matrix[k, i, j-1])
        return cummulative_matrix

    def optimal_path(self, cummulative_matrix):
        '''
        This function will find an optimal path of a given DTW matrix
          :param cummulative_matrix, DTW cost matrix, shape batch x n x m

          :return optimal path of the dtw matrix
        '''
        B, N, M = cummulative_matrix.shape
        N, M = N-1, M-1
        _P = []
        for b in range(B):
            p = [(N, M)]
            matrix = cummulative_matrix[b, ...]
            while N>0 or M>0:
                if N == 0:
                  cell = (N, M-1)
                elif M==0:
                  cell = (N-1, M)
                else:
                  _min = min(matrix[N-1, M-1],
                             matrix[N-1, M],
                             matrix[N, M-1])
                  if _min == matrix[N-1, M-1]: cell = (N-1, M-1)
                  elif _min == matrix[N, M-1]: cell = (N, M-1)
                  else: cell = (N-1, M)
                p.append(cell)
                N, M = cell
            p.reverse()
            _P.append(p)
        return _P

    def allignment(self, keypoint_batch):
        '''
        batch: number of states (or timeseries)
        17: num of keypoints for each pose
        3: [x, y, z]
          :param keypoint_batch, batch x 17 x 3
        '''
        def proj(self, keypoints, projection = 'XOY'):
            '''
            This function will translation all points to new coordinate system
            17 keyspoints x [x, y, z]

              :param vetor, shape 17 x 3
              :param projection: projection space
            '''
            transformation_matrix = np.eye(3) # shape 3 x 3
            A, B, C = keypoints[0], keypoints[1], keypoints[4] # each has shape [3, ]
            AB, AC = B - A, C - A
            xAB, yAB = AB[0], AB[1]
            xAC, yAC = AC[0], AC[1]
            idx, idy = (0, 1, 0, 1), (0, 0, 1, 1)
            transformation_matrix[idx, idy] = xAB, yAB, xAC, yAC # Ox => BA, Oy => AC
            new_matrix = transformation_matrix.dot(keypoints.T) # shape 3 x 17
            return new_matrix.T # 17 x 3

        b, nums, depth = keypoint_batch.shape
        new_batch = np.empty(shape = (b, nums, depth))
        for i in range(b):
            new_batch[i] = self.proj(keypoint_batch[i])
        return new_batch

    def euler_rotation(self, in_matrix, angle, axis = 'Z', is_radian = False):
        '''
        This function will rotate a plane along its given orthogonal axis
        17 x 3: 17 jointed-points x [x, y, z]
          :param in_matrix
          :return 17 x 3
        '''
        assert axis in ['X', 'Y', 'Z']
        a = angle if is_radian else angle / 180 * np.pi
        rotate_matrix = np.eye(3)
        if axis == 'Z':
          y, x = ((0, 0, 1, 1), (0, 1, 0, 1))
        elif axis == 'Y':
          y, x = ((0, 2, 0, 2), (0, 0, 2, 2))
        elif axis == 'X':
          y, x = ((1, 1, 2, 2), (1, 2, 1, 2))

        rotate_matrix[y, x] = np.cos(a), -np.sin(a), np.sin(a), np.cos(a)
        new = rotate_matrix.dot(in_matrix.T)
        return new.T

    def point_to_vec(self, matrix):
        '''
        This function transform list of point to list of desired vector
        batch: nums of states
        n: 17 keypoints
        d: [x, y, z]
          :param matrix: batch x n x d

          :return vec: batch x n-1 x d
        '''
        def _vec(matrix, x, y):
            return matrix[:, y:y+1, :] - matrix[:, x:x+1, :]

        pair = [(0, 1), (0, 4), (1, 2), (2, 3), (4, 5), (5, 6), (7, 0), (8, 7), 
                (8, 14), (14, 15), (15, 16), (8, 11), (11 ,12), (12, 13), (9, 8), (10, 9)]
        vec = [_vec(matrix, i, j) for (i, j) in pair] # shape of one element batch x 1 x 3
        vec = np.concatenate(vec, axis = 1)
        return vec

    def compare_1_1(self, gt, pred, _type = 'rotate'):
        '''
        Compare 
          :param gt: nums_state x 17 x 3
          :param pred: nums_state x 17 x 3
          :param type: ['rotate', 'allign']

          :return points between each state
        '''
        b = gt.shape[0]
        if _type == 'rotate': 
            pa = []
            _gt = self.point_to_vec(gt) # shape 8 x 16 x 3
            for j in range(b):   
                pb = []
                for i in range(360):
                    _pred = self.euler_rotation(pred[j], i)
                    _pred = self.point_to_vec(_pred[np.newaxis, ...]) # shape 8 x 16 x 3     
                    GT = _gt[j] # 16 x 3 
                    PRED = _pred[0] # 16 x 3
                    mul = (GT**2).sum(axis = 1) * (PRED**2).sum(axis = 1)
                    cos = (GT*PRED).sum(axis = 1) / mul**0.5 # shape = (16, )
                    pb.append(np.arccos(cos))
                pa.append(pb)
            pa = np.array(pa) # b x 360 x 16
            pa = pa.mean(axis = -1)
            return pa.min(axis = 1), pa.argmin(axis = 1) # b x 16

        elif _type == 'allign':
            pb = []
            _gt = self.point_to_vec(self.allignment(gt))
            _pred = self.point_to_vec(self.allignment(pred)) # shape 8 x 16 x 3
            pb = []
            for j in range(b):
                GT = _gt[j]
                PRED = _pred[j] # 16 x 3
                mul = (GT**2).sum(axis = 1) * (PRED**2).sum(axis = 1)
                cos = (GT*PRED).sum(axis = 1) / mul**0.5 # shape = (16, )
                pb.append(np.arccos(cos))   
            return np.array(pb)

    def compare(self):
        raise NotImplementedError

    def scoring(self):
        raise NotImplementedError

    def __call__(self):
        raise NotImplementedError
