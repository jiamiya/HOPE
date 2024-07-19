
import pickle
import time

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randn, random
from shapely.geometry import LinearRing
from typing import List
import cv2

from .map_level import get_map_level
from .vehicle import State
from .map_base import *
from configs import *

class ParkingMapGrid(object):
    default = {
        'data_dir': '../data/grid_map',
    }
    def __init__(self, xy_reoslution=0.1):

        self.case_id:int = -1
        self.start: State = None
        self.dest: State = None
        self.start_box:LinearRing = None
        self.dest_box:LinearRing = None
        self.xmin, self.xmax = 0, 0
        self.ymin, self.ymax = 0, 0
        self.grid_map = None
        self.origin_roi = None
        self.xy_resolution = xy_reoslution
        self._vehicle_box_grid = self._init_vehicle_box_grid()
        self._init_map()

    def _init_map(self,):
        # load the files of grid_map
        if os.path.exists(self.default['data_dir']):
            data_files = os.listdir(self.default['data_dir'])
            data_files.sort()
            self.map_data = []
            for data_file in data_files:
                with open(os.path.join(self.default['data_dir'], data_file), 'rb') as f:
                    data = pickle.load(f)
                    self.map_data.append(data)

        self.case_ids = np.arange(len(self.map_data))



    def rear_center_to_parkinglot_center(self, rear_center_x, rear_center_y, heading):
        shift_distance = LENGTH / 2 - REAR_HANG - 0.4
        center_x = rear_center_x + shift_distance * np.cos(heading)
        center_y = rear_center_y + shift_distance * np.sin(heading)
        return center_x, center_y, heading
    
    def parkinglot_center_to_rear_center(self, center_x, center_y, heading):
        shift_distance = LENGTH / 2 - REAR_HANG - 0.4
        rear_center_x = center_x - shift_distance * np.cos(heading)
        rear_center_y = center_y - shift_distance * np.sin(heading)
        return rear_center_x, rear_center_y, heading

    def enhance_map_virtual_vehicle(self, vehicle_pose:tuple):
        h, w = np.random.randint(16, 24), np.random.randint(42, 50)
        vehicle_img = np.ones((h, w), dtype=np.uint8) #*255
        x, y, yaw = vehicle_pose
        x += np.random.uniform(-0.1, 0.1)
        y += np.random.uniform(-0.1, 0.1)
        yaw += np.random.uniform(-np.pi/36, np.pi/36)

        x, y, yaw = self.rear_center_to_parkinglot_center(x, y, yaw)
        x, y = self.coord2grid(np.array([[x, y]]).astype(float).reshape(-1, 2)).squeeze()
        x, y = int(x), int(y)
        center_x, center_y = w // 2, h // 2

        def get_affine_transform_matrix(x, y, yaw, x1, y1, yaw1):
            dx = x1
            dy = y1
            d_yaw = yaw1 - yaw
            translation_matrix1 = np.array([[1, 0, -x],
                                        [0, 1, -y],
                                        [0, 0, 1]], dtype=np.float32)
            rotation_matrix = np.array([[np.cos(d_yaw), -np.sin(d_yaw), 0],
                                            [np.sin(d_yaw), np.cos(d_yaw), 0],
                                            [0, 0, 1]], dtype=np.float32)
            translation_matrix2 = np.array([[1, 0, dx],
                                        [0, 1, dy],
                                        [0, 0, 1]], dtype=np.float32)

            affine_matrix = translation_matrix2@ rotation_matrix@ translation_matrix1
            return affine_matrix[:2]
        
        M = get_affine_transform_matrix(center_x, center_y, 0, x, y, yaw)

        rotated_small_img = cv2.warpAffine(vehicle_img, M, self.grid_map.shape)

        self.grid_map = np.clip(self.grid_map + rotated_small_img, 0, 1)


    def reset(self, case_id:int, data_dir:str=None):
        case_id = self.case_ids[case_id % len(self.case_ids)]
        data = self.map_data[case_id]
        self.grid_map = data['gridmap']
        self.origin_roi = data['map_range']
        self.xmin = self.origin_roi[0]
        self.xmax = self.origin_roi[1]
        self.ymin = self.origin_roi[2]
        self.ymax = self.origin_roi[3]
        self.starts = data['starts']
        start = self.starts[np.random.randint(len(self.starts))]
        self.xy_resolution = data['xy_resolution']
        dest = data['goal']

        self.start = State(start)
        self.dest = State(dest)
        self.start_box = self.start.create_box()
        self.dest_box = self.dest.create_box()

        return self.start
        

    def grid2coord(self, grids:np.ndarray):
        # grids: (n, 2)
        x, y = grids[:, 0], grids[:, 1]
        return np.array([x*self.xy_resolution+self.xmin, y*self.xy_resolution+self.ymin], dtype=float).transpose()
    
    def _init_vehicle_box_grid(self):
        coords = np.array(VehicleBox.coords)[:4]
        x, y = coords[:, 0], coords[:, 1]
        return np.array([(x-0)/self.xy_resolution, (y-0)/self.xy_resolution], dtype=float).transpose()

    def coord2grid(self, coords:np.ndarray, remain_float=False):
        # coords: (n, 2)
        x, y = coords[:, 0], coords[:, 1]
        if remain_float:
            return np.array([(x-self.xmin)/self.xy_resolution, (y-self.ymin)/self.xy_resolution], dtype=float).transpose()
        else:
            return np.array([(x-self.xmin)/self.xy_resolution, (y-self.ymin)/self.xy_resolution], dtype=int).transpose()
        
    
    def collision_ckeck(self, trajs:list):
        background = np.zeros((self.grid_map.shape[0], self.grid_map.shape[1]), dtype=np.uint8)
        grid_map_flip = self.grid_map.copy()
        vehicle_positions = np.array(trajs)  # (n, 3)
        vehicle_box_coords = np.stack([np.array(State(trajs[i]).create_box().coords)[:4] for i in range(len(trajs))], axis=0)
        vehicle_box_grid_coords = self.coord2grid(vehicle_box_coords.reshape(-1, 2)).reshape(-1, 4, 2)
        for i in range(vehicle_positions.shape[0]):
            rotated_points = vehicle_box_grid_coords[i]
            cv2.fillPoly(background, [rotated_points], 255)
        if np.any(background & (grid_map_flip != 0)):
            return True
        else:
            return False
        

        