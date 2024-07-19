import sys
sys.path.append("../")
sys.path.append(".")
import math
import numpy as np
from shapely.geometry import LineString, Point
from shapely.affinity import affine_transform

from env.vehicle import State,VehicleBox
from env.parking_map_grid import ParkingMapGrid

ORIGIN = Point((0,0))

class LidarSimlator():
    def __init__(self, 
        lidar_range:float = 10.0,
        lidar_num:int = 120
    ) -> None:
        '''
        Args:
            lidar_range(float): the max distance that the obstacle can be dietected.
            lidar_num(int): the beam num of the lidar simulation.
        '''
        self.lidar_range = lidar_range
        self.lidar_num = lidar_num
        self.lidar_lines = []
        for a in range(lidar_num):
            self.lidar_lines.append(LineString(((0,0), (math.cos(a*math.pi/lidar_num*2)*lidar_range,\
                 math.sin(a*math.pi/lidar_num*2)*lidar_range))))
        self.vehicle_boundary = self.get_vehicle_boundary()
        self.lidar_tolerance = 0.1

    def get_observation(self, ego_state:State, obstacles):
        '''
        Get the lidar observation from the vehicle's view.

        Args:
            ego_state: the state of ego car.
            obstacles: the list of obstacles in map, or the image of obstacles grid map.

        Return:
            lidar_obs(np.array): the lidar data in sequence of angle, with the length of lidar_num.
        '''
        if isinstance(obstacles, list):
            ego_pos = (ego_state.loc.x, ego_state.loc.y, ego_state.heading)
            rotated_obstacles = self._rotate_and_filter_obstacles(ego_pos, obstacles)
            lidar_obs = self._fast_calc_lidar_obs(rotated_obstacles)
        else:
            lidar_obs = self._fast_calc_lidar_obs_on_grid(ego_state.get_pos(), obstacles)
        lidar_obs = np.array(lidar_obs - self.vehicle_boundary)
        lidar_obs = np.clip(lidar_obs-self.lidar_tolerance, 1e-4, self.lidar_range)
        return lidar_obs
    
    
    def _fast_calc_lidar_obs_on_grid(self, ego_pos:tuple, grid_map:ParkingMapGrid):
        x,y,yaw = ego_pos
        lidar_range = self.lidar_range
        xy_resolution = grid_map.xy_resolution
        step_num = int(lidar_range/xy_resolution)
        angles = yaw + np.arange(0, 2*np.pi, 2*np.pi/self.lidar_num) # (120,)
        angles = angles.reshape(-1, 1) # (120,1)
        steps = np.linspace(xy_resolution, lidar_range, step_num) # (1, 100)
        pts_x = x + steps*np.cos(angles) # (120, 100)
        pts_y = y + steps*np.sin(angles) # (120, 100)
        pts = np.stack([pts_x, pts_y], axis=-1) # (120, 100, 2)
        pts_coords = grid_map.coord2grid(pts.reshape(-1, 2)) # (120*100, 2) # .reshape(-1, step_num, 2)
        # pts_coords = np.round(pts/xy_resolution).astype(np.int) # (120, 100, 2)
        # transpose x and y, (n,2)->(n,2)
        pts_coords = np.array([pts_coords[:,1], pts_coords[:,0]]).T.reshape(-1, step_num, 2) # (120, 100, 2)
        pts_coords_outrange_x = np.logical_or(pts_coords[:,:,0]<0, pts_coords[:,:,0]>=grid_map.grid_map.shape[0]) # (120, 100)
        pts_coords_outrange_y = np.logical_or(pts_coords[:,:,1]<0, pts_coords[:,:,1]>=grid_map.grid_map.shape[1]) # (120, 100)
        pts_coords_outrange = np.logical_or(pts_coords_outrange_x, pts_coords_outrange_y) # (120, 100)
        # tmperarily set the out of range points to the (0,0)
        pts_coords[pts_coords_outrange] = np.array([0,0])
        is_occupied = grid_map.grid_map[pts_coords[:,:,0], pts_coords[:,:,1]] # (120, 100)
        # recover the out of range points to the max distance
        is_occupied[pts_coords_outrange] = 0
        is_occupied[:, -1] = 1
        min_dist_idx = np.argmax(is_occupied, axis=1) # (120,)
        lidar_obs = steps[min_dist_idx] # (120,)

        return lidar_obs

    def get_vehicle_boundary(self, ):
        lidar_base = []
        for l in self.lidar_lines:
            distance = l.intersection(VehicleBox).distance(ORIGIN)
            lidar_base.append(distance)
        return np.array(lidar_base)

    def _rotate_and_filter_obstacles(self, ego_pos:tuple, obstacles:list):
        '''
        Rotate the obstacles around the vehicle and remove the obstalces which is out of lidar range.
        '''
        x, y, theta = ego_pos
        a = math.cos(theta)
        b = math.sin(theta)
        x_off = -x*a - y*b
        y_off = x*b - y*a
        affine_mat = [a, b, -b, a, x_off, y_off]

        rotated_obstacles = []
        for obs in obstacles:
            rotated_obs = affine_transform(obs, affine_mat)
            if rotated_obs.distance(ORIGIN) < self.lidar_range:
                rotated_obstacles.append(rotated_obs)
        
        return rotated_obstacles
    
    def _fast_calc_lidar_obs(self, obstacles:list):
        '''
        Obtain the lidar observation making use of numpy builtin matrix acceleration.

        Parameter:
            obstacles ( list(LinearRing) ): the obstacles around the vehicle which have been transformed to the ego referrence.

        Return:
            lidar_obs (np.ndarray): in shape (LIDAR_NUM,)
        '''

        # Line 1: the lidar ray, ax + by + c = 0
        theta = np.array([a*math.pi/self.lidar_num*2 for a in range(self.lidar_num)]) # (120,)
        a = np.sin(theta).reshape(-1,1) # (120, 1)
        b = -np.cos(theta).reshape(-1,1)
        c = 0

        # convert obstacles(LinerRing) to edges ((x1,y1), (x2,y2))
        x1s, x2s, y1s, y2s = [], [], [], []
        for obst in obstacles:
            obst_coords = np.array(obst.coords) # (n+1,2)
            x1s.extend(list(obst_coords[:-1, 0]))
            x2s.extend(list(obst_coords[1:, 0]))
            y1s.extend(list(obst_coords[:-1, 1]))
            y2s.extend(list(obst_coords[1:, 1]))
        if len(x1s) == 0: # no obstacle around
            return np.ones((self.lidar_num))*self.lidar_range
        x1s, x2s, y1s, y2s  = np.array(x1s).reshape(1,-1), np.array(x2s).reshape(1,-1),\
            np.array(y1s).reshape(1,-1), np.array(y2s).reshape(1,-1), 
        # Line 2: the edges of obstacles, dx + ey + f = 0
        d = (y2s - y1s).reshape(1,-1) # (1,E)
        e = (x1s - x2s).reshape(1,-1)
        f = (y1s*x2s - x1s*y2s).reshape(1,-1)

        # calculate the intersections
        det = a*e - b*d # (120, E)
        parallel_line_pos = (det==0) # (120, E)
        det[parallel_line_pos] = 1 # temporarily set "1" to avoid "divided by zero"
        raw_x = (b*f - c*e)/det # (120, E)
        raw_y = (c*d - a*f)/det

        # select the true intersections, set the false positive interesections to inf
        tmp_inf = 100
        tmp_zero = 1e-8
        # the false positive intersections on line L1(not on ray L1)
        # here we assume the orientation of lidar[0] is 0 rad in the ego coordinate.
        raw_x[:self.lidar_num//4][raw_x[:self.lidar_num//4]<-tmp_zero] = tmp_inf
        raw_x[self.lidar_num//4*3:][raw_x[self.lidar_num//4*3:]<-tmp_zero] = tmp_inf
        raw_x[self.lidar_num//4:self.lidar_num//4*3][raw_x[self.lidar_num//4:self.lidar_num//4*3]>tmp_zero] = tmp_inf
        raw_y[:self.lidar_num//2][raw_y[:self.lidar_num//2]<-tmp_zero] = tmp_inf
        raw_y[self.lidar_num//2:][raw_y[self.lidar_num//2:]>tmp_zero] = tmp_inf
        # the false positive intersections on line L2(not on edge L2)
        raw_x[raw_x>np.maximum(x1s, x2s)] = tmp_inf
        raw_x[raw_x<np.minimum(x1s, x2s)] = tmp_inf
        raw_y[raw_y>np.maximum(y1s, y2s)] = tmp_inf
        raw_y[raw_y<np.minimum(y1s, y2s)] = tmp_inf
        # the (L1, L2) which are parallel
        raw_x[parallel_line_pos] = tmp_inf

        lidar_obs = np.min(np.sqrt(raw_x**2 + raw_y**2), axis=1) # (120,)
        lidar_obs = np.clip(lidar_obs, 0, self.lidar_range)
        return lidar_obs


