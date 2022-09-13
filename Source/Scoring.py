import numpy as np
import math
import numba
import json
from tqdm import tqdm 
from collections import defaultdict

class scoring():
    def __init__(self, method = 'cosin'):
        assert method in ['cosin', 'L1']
        self.method = method
        self.cost_func = lambda x, y: self.cosin(x, y) \
              if self.method == 'cosin' else np.abs(x - y)

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
        batch: num of jonit vectors
          :param cummulative_matrix, DTW cost matrix, shape batch x n x m

          :return optimal path of the dtw matrix
        '''
        B, n, m = cummulative_matrix.shape
        _P = []
        _V = []
        for b in range(B):
            N, M = n-1, m-1
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
            idx = tuple(zip(*p))
            pvalue = cummulative_matrix[b, ...][idx]
            _P.append(p)
            _V.append(pvalue)
        return _P, _V


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


    def euler_rotation(self, in_matrix, angle, axis = 'Z', is_radian = False):
        '''
        This function will rotate a plane along its given orthogonal axis
        17 x 3: 17 jointed-points x [x, y, z]
          :param in_matrix, 17 x 3
          :return set of rotated keypoints with shape: 17 x 3
        '''
        assert axis in ['X', 'Y', 'Z']
        a = angle if is_radian else angle / 180 * np.pi
        rotate_matrix = np.eye(3) # shape 3 x 3 
        if axis == 'Z':
            y, x = ((0, 0, 1, 1), (0, 1, 0, 1))
        elif axis == 'Y':
            y, x = ((0, 2, 0, 2), (0, 0, 2, 2))
        elif axis == 'X':
            y, x = ((1, 1, 2, 2), (1, 2, 1, 2))

        rotate_matrix[y, x] = np.cos(a), -np.sin(a), np.sin(a), np.cos(a)
        new = rotate_matrix.dot(in_matrix.T) # shape 3 x 17
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

        part_pair_dict = {
            'left hand': [(11, 12), (12, 13)],
            'right hand': [(14, 15), (15, 16)],
            'left leg': [(4, 5), (5, 6)],
            'right leg': [(1, 2), (2, 3)],
            'spine': [(10, 9), (9, 8), (8, 7), (7,0)],
            'shoulder': [(8, 11), (8, 14)],
            'pelvis': [(0, 1), (0, 4)]
        }

        vec = [_vec(matrix, i, j) for (i, j) in pair] # shape of one element: batch x 1 x 3
        vec = np.concatenate(vec, axis = 1)
        
        part_vec = {}
        part_vec.update({key:np.concatenate(
                              list(map(lambda x: _vec(matrix, x[0], x[1]), part_pair_dict[key])), 
                              axis = 1) for key in part_pair_dict.keys()
                        })
        return vec, part_vec


    def compare_1_1(self, gt, pred):
        '''
        Compare each state
        num_states: number of states used
        17 keypoints
        3: [x, y, z]
          :param gt: 1 x 17 x 3
          :param pred: 1 x 17 x 3

          :return points between each state
        '''
        assert gt.shape[0] == pred.shape[0], \
        print(f'Number of states of gt and pred have to be the same\n \
                batch of gt {gt.shape[0]} \
                batch of pred {pred.shape[0]}')

        b = gt.shape[0]
        pa = {}
        _gt, _gt_dict = self.point_to_vec(gt) # shape num_states x 16 x 3
        for key in _gt_dict: 
            pb = []  
            for j in range(b):
                _min = None
                for i in range(360):
                    _pred = self.euler_rotation(pred[j], i)
                    _pred, _pred_dict = self.point_to_vec(_pred[np.newaxis, ...]) # shape 1 x 16 x 3   
                                                                                  # dict 5 parts 1 x num_vecs x 3  
                    GT = _gt_dict[key][j] # num_vecs x 3 
                    PRED = _pred_dict[key][0] # num_vecs x 3
                    mul = (GT**2).sum(axis = 1) * (PRED**2).sum(axis = 1)
                    cos = (GT*PRED).sum(axis = 1) / mul**0.5 # shape = (num_vecs, )
                    tmp = np.mean(np.arccos(cos))
                    if _min is None:
                        _min, argmin = tmp, i
                        argcos = np.arccos(cos)
                    else:
                        if tmp < _min:
                            _min, argmin = tmp, i
                            argcos = np.arccos(cos)
                pb.append(argcos)           
            pa[key] = np.array(pb) # (num_vecs, )
        return pa

    def compare_dtw(self, gt, pred):
        _dtw = {}
        _min = None
        _gt_vecs, _gt_parts = self.point_to_vec(gt)
        for i in range(360):
            _pred = np.empty(pred.shape)
            for j in range(pred.shape[0]):
                _pred[j] = self.euler_rotation(pred[j], i)
            
            _pred_vecs, _pred_parts = self.point_to_vec(_pred)
            dtw_matrix = self.dtw(_gt_vecs.transpose(1, 2, 0), _pred_vecs.transpose(1, 2, 0))
            _, value = self.optimal_path(dtw_matrix)
            point = list(map(lambda x: x[-1], value))
            point = np.array(point).mean()

            if _min is None or _min > point:
                _min = point
                _lambda = lambda x, y: self.optimal_path(self.dtw(x, y))[1]
                _dtw.update({key:list(map(lambda x: (x[-1], x.shape[0]), 
                                          _lambda(_gt_parts[key].transpose(1, 2, 0), # shape 16 x 3 x num_vecs
                                                  _pred_parts[key].transpose(1, 2, 0)))) for key in _gt_parts.keys()
                            })
        return _dtw

    def scoring(self, raw):
        '''
          :param raw: dict of part body vecs
        '''
        _raw = {}
        # _raw.update({key: 100 - raw[key].mean(axis=-1) / np.pi * 100 for key in raw})
        _raw.update({key: (1 - raw[key] / np.pi)**(1.5) * 100 for key in raw})
        return _raw


    def __call__(self, gt, pred, _type = '1v1', allignment = 'allign', reduction = 'mean'):
          '''
          Run
          batch: time series for 'series' or nums of states for '1v1'
          nums joints: number of keypoint used, 17
          3: [x ,y, z]
            :param gt, batch x 17 x 3
            :param pred, batch x 17 x 3
            :param _type
            :param allignment
            :param reduction
          '''
          assert reduction in [None, 'mean']
          if _type == '1v1':
              raw = self.compare_1_1(gt, pred)
          elif _type == 'series':
              raw = self.compare_DTW(gt, pred, allignment)
              raw.update({ key: np.array(list(map(lambda x: x[0]/x[1], raw[key]))) for key in raw})             

          # if reduction == 'mean':
          #     raw = raw.mean(axis = 1)
                  
          return self.scoring(raw), raw
