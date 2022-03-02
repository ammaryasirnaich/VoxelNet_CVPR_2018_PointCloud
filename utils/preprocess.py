#!/usr/bin/env python
# -*- coding:UTF-8 -*-

# File Name : preprocess.py
# Purpose :
# Creation Date : 18-02-2022
# Last Modified : Thu 18 Jan 2018 05:34:42 PM CST
# Created By : Ammar Yasir Naich [ammar[dot]naich[at]gmail[dot]com]


from ctypes import util
import numpy as np
from pyrsistent import v
from config import cfg
import utils
from utils.voxel_generator import *
from utils import intensity_histogram 




data_dir = 'velodyne'

def vfe_from_pointcloud(point_cloud, cls = cfg.DETECT_OBJ):
    
    """ Description
    :type point_cloud: ndarray [M,4]
    :param point_cloud:
    
    :type cls: string
    :param cls: object names " Car, Pedestrian, Cyclist
    
    :rtype: dictionary containing keys for 'feature buffer', coordinate_buffer, 'number_buffer'

    """
    
    # voxel_size = [2, 2, 4]
    # point_cloud_range = [0, -40, -3, 70.4, 40, 1]
    voxel_size = [0.2, 0.2, 0.4]
    point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1] # config paraa taken from pointpilleer
    max_num_points = 35
    self = VoxelGenerator(voxel_size, point_cloud_range, max_num_points)
    voxels = self.generate(point_cloud)
    voxels, voxel_index, num_points_per_voxel = voxels
    
    
    # [K, 3] coordinate buffer as described in the paper, is the non-emppty voxels index
    coordinate_buffer = np.unique(voxel_index, axis = 0)

    K = len(coordinate_buffer) # number of voxels
    T = max_num_points

    # [K, 1] store number of points in each voxel grid
    number_buffer = np.zeros(shape = (K), dtype = np.int64)

    # [K, T, 7] feature buffer as described in the paper
    feature_buffer = np.zeros(shape = (K, T, 7), dtype = np.float32)
        
    print("Debug info about accelerated method for voxel generator")
    print(self)
    print("Selecting the dynamic grid size", self.grid_size)
    print(f'Printing pointclouud shape',voxels.shape)
    print(f'Printing feature voxel_index shape',voxel_index.shape)
    print(f'Printing K value', K)
    print("feature buffer shape", feature_buffer.shape)
    

    # build a reverse index for coordinate buffer
    index_buffer = {}
    for i in range(K):
        index_buffer[tuple(coordinate_buffer[i])] = i

    for voxel, point in zip(voxel_index, point_cloud):
        index = index_buffer[tuple(voxel)]
        number = number_buffer[index]
        if number < T:
            feature_buffer[index, number, :4] = point
            number_buffer[index] += 1

    feature_buffer[:, :, -3:] = feature_buffer[:, :, :3] - \
        feature_buffer[:, :, :3].sum(axis = 1, keepdims = True)/number_buffer.reshape(K, 1, 1)

    voxel_dict = {'feature_buffer': feature_buffer,
                  'coordinate_buffer': coordinate_buffer,
                  'number_buffer': number_buffer}

    return voxel_dict


def process_pointcloud(point_cloud, cls = cfg.DETECT_OBJ):
    # Input:
    #   (N, 4)
    # Output:
    #   voxel_dict
    if cls == 'Car':
        scene_size = np.array([4, 80, 70.4], dtype = np.float32)
        voxel_size = np.array([0.4, 0.2, 0.2], dtype = np.float32)
        grid_size = np.array([10, 400, 352], dtype = np.int64)
        lidar_coord = np.array([0, 40, 3], dtype = np.float32)
        max_point_number = 35
    else:
        scene_size = np.array([4, 40, 48], dtype = np.float32)
        voxel_size = np.array([0.4, 0.2, 0.2], dtype = np.float32)
        grid_size = np.array([10, 200, 240], dtype = np.int64)
        lidar_coord = np.array([0, 20, 3], dtype = np.float32)
        max_point_number = 45

        np.random.shuffle(point_cloud)

    shifted_coord = point_cloud[:, :3] + lidar_coord
    # reverse the point cloud coordinate (X, Y, Z) -> (Z, Y, X)
    voxel_index = np.floor(
        shifted_coord[:, ::-1] / voxel_size).astype(np.int)

    bound_x = np.logical_and(
        voxel_index[:, 2] >= 0, voxel_index[:, 2] < grid_size[2])
    bound_y = np.logical_and(
        voxel_index[:, 1] >= 0, voxel_index[:, 1] < grid_size[1])
    bound_z = np.logical_and(
        voxel_index[:, 0] >= 0, voxel_index[:, 0] < grid_size[0])

    bound_box = np.logical_and(np.logical_and(bound_x, bound_y), bound_z)

    point_cloud = point_cloud[bound_box]
    voxel_index = voxel_index[bound_box]

    # [K, 3] coordinate buffer as described in the paper
    coordinate_buffer = np.unique(voxel_index, axis = 0)

    K = len(coordinate_buffer)
    T = max_point_number

    # [K, 1] store number of points in each voxel grid
    number_buffer = np.zeros(shape = (K), dtype = np.int64)

    # [K, T, 7] feature buffer as described in the paper
    # feature_buffer = np.zeros(shape = (K, T, 7), dtype = np.float32)
    
    # extending feature  duffer to include intensity signature
    feature_buffer = np.zeros(shape = (K, T, 7), dtype = np.float32)

    # print("Debug info about author method for voxel generator")
    # print(f'Printing pointclouud shape',point_cloud.shape)
    # print(f'Printing feature voxel_index shape',voxel_index.shape)
    # print(f'Printing K value', K)
    # print("feature buffer shape", feature_buffer.shape)
   
    # build a reverse index for coordinate buffer
    index_buffer = {}
    for i in range(K):
        index_buffer[tuple(coordinate_buffer[i])] = i

    for voxel, point in zip(voxel_index, point_cloud):
        index = index_buffer[tuple(voxel)]
        number = number_buffer[index]
        if number < T:
            feature_buffer[index, number, :4] = point
            number_buffer[index] += 1


    
    feature_buffer[:, :, -3:] = feature_buffer[:, :, :3] - \
        feature_buffer[:, :, :3].sum(axis = 1, keepdims = True)/number_buffer.reshape(K, 1, 1)
    
    
    
    
        
    # n_bin = 10
    # intensity_values = feature_buffer[:, :, 3]
    
    # min_int_values = np.min(intensity_values)
    # max_int_values = np.max(intensity_values)
    # print("Overall feature dimensions",feature_buffer.shape)
    # print("Number Buffer dimensions",number_buffer.shape )
    # print("intensity_values shape",intensity_values.shape)
    # print("min_int_values",min_int_values)
    # print("max_int_values,max_int_values")
    # hist,bin_edge = intensity_histogram.numba_gpu_histogram(feature_buffer[:, :, 3],n_bin)
    # intensity_feature = intensity_histogram.generate_hist_feature(hist,bin_edge[:-1])
    
    # print("feature Dtype",type(intensity_feature))
    # print("feature shape",intensity_feature.shape)

    voxel_dict = {'feature_buffer': feature_buffer,
                  'coordinate_buffer': coordinate_buffer,
                  'number_buffer': number_buffer}

    return voxel_dict






