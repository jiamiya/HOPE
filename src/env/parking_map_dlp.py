
import pickle

import numpy as np
from numpy.random import randn, random
from shapely.geometry import LinearRing
from typing import List

from env.map_level import get_map_level
from env.vehicle import State
from env.map_base import *
from configs import *


class ParkingMapDLP(object):
    default = {
        'path': '../data/dlp.data'
    }
    def __init__(self):

        self.case_id:int = None
        self.start: State = None
        self.dest: State = None
        self.start_box:LinearRing = None
        self.dest_box:LinearRing = None
        self.xmin, self.xmax = 0, 0
        self.ymin, self.ymax = 0, 0
        self.n_obstacle = 0
        self.obstacles:List[Area] = []

        f_map = open(self.default['path'], 'rb')
        self.map_data = pickle.load(f_map) # n* (start_candidates, dest, obstacles, traj_path)
        f_map.close()
        self.multi_start = False
        if isinstance(self.map_data[0][0], list):
            self.multi_start = True

    def reset(self, case_id: int = None, path: str = None) -> State:
        if path is not None:
            f_map = open(path, 'rb')
            self.map_data = pickle.load(f_map)
            f_map.close()
            if isinstance(self.map_data[0][0], list):
                self.multi_start = True
            else:
                self.multi_start = False

        if case_id is None:
            self.case_id = np.random.randint(0, len(self.map_data))
        else:
            if case_id >= len(self.map_data):
                case_id = case_id%len(self.map_data)
            self.case_id = case_id
        start, dest, obstacles = self.map_data[self.case_id][:3]
        # if len(self.map_data[self.case_id]) == 4:
        #     self.traj_path = self.map_data[self.case_id][3]
        if isinstance(start, tuple):
            start = list(start)
        if isinstance(dest, tuple):
            dest = list(dest)
        
        if self.multi_start:
            random_id = np.random.randint(0, len(start))
            start = start[random_id]
            start = (start[0] + randn()*0.05, start[1] + randn()*0.05, start[2] + randn()*0.02)
        self.start = State(start)
        self.start_box = self.start.create_box()
        self.dest = State(dest)
        self.dest_box = self.dest.create_box()
        self.xmin = np.floor(min(self.start.loc.x, self.dest.loc.x) - 20)
        self.xmax = np.ceil(max(self.start.loc.x, self.dest.loc.x) + 20)
        self.ymin = np.floor(min(self.start.loc.y, self.dest.loc.y) - 20)
        self.ymax = np.ceil(max(self.start.loc.y, self.dest.loc.y) + 20)
        if not isinstance(obstacles[0], LinearRing):
            raise UserWarning('Obstcale shape should be shapely.LinearRing !')
        self.obstacles = list([Area(shape=obs, subtype="obstacle", \
            color=(150, 150, 150, 255)) for obs in obstacles])
        self.n_obstacle = len(self.obstacles)
        self.filter_obstacles()
        if random() > 0.5:
            self.flip_dest_orientation()
        if random() > 0.5:
            self.flip_start_orientation()
        self.map_level = get_map_level(self.start, self.dest, self.obstacles)

        return self.start
    
    def filter_obstacles(self,):
        filtered_obstacles = []
        for obs_area in self.obstacles:
            obst_coords = np.array(obs_area.shape.coords)
            x_max = np.max(obst_coords[:,0])
            x_min = np.min(obst_coords[:,0])
            y_max = np.max(obst_coords[:,1])
            y_min = np.min(obst_coords[:,1])
            if not (x_max <= self.xmin or x_min >= self.xmax or\
            y_max <= self.ymin or y_min >= self.ymax):
                filtered_obstacles.append(obs_area)
        self.obstacles = filtered_obstacles
        self.n_obstacle = len(self.obstacles)
        return

    def get_boundary(self,):
        obst_coords = []
        for obs in self.obstacles:
            obst_coords.extend(list(obs.shape.coords))
        obst_coords = np.array(obst_coords)
        self.xmin = np.floor(min(np.min(obst_coords[:,0]), self.xmin))
        self.xmax = np.floor(max(np.max(obst_coords[:,0]), self.xmax))
        self.ymin = np.floor(min(np.min(obst_coords[:,1]), self.ymin))
        self.ymax = np.floor(max(np.max(obst_coords[:,1]), self.ymax))
    
    def change_start_dest(self,):
        self.start, self.dest = self.dest, self.start
        self.start_box, self.dest_box = self.dest_box, self.start_box

    def _flip_box_orientation(self, target_state:State):
        x, y, heading = target_state.get_pos()
        center = np.mean(target_state.create_box().coords[:-1], axis=0)
        new_x = 2*center[0] - x
        new_y = 2*center[1] - y
        heading = heading + np.pi
        return State([new_x, new_y, heading])
    
    def flip_dest_orientation(self,):
        self.dest = self._flip_box_orientation(self.dest)
        self.dest_box = self.dest.create_box()

    def flip_start_orientation(self,):
        self.start = self._flip_box_orientation(self.start)
        self.start_box = self.start.create_box()