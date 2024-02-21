import sys
sys.path.append("../")
sys.path.append(".")
import math
import time
import numpy as np
from shapely.geometry import LineString, Point
from shapely.affinity import affine_transform

from env.vehicle import State,VehicleBox

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
        self.vehicle_base = self.get_vehicle_base()

    def get_observation(self, ego_state:State, obstacles:list):
        '''
        Get the lidar observation from the vehicle's view.

        Args:
            ego_state: the state of ego car.
            obstacles: the list of obstacles in map

        Return:
            lidar_obs(np.array): the lidar data in sequence of angle, with the length of lidar_num.
        '''

        ego_pos = (ego_state.loc.x, ego_state.loc.y, ego_state.heading)
        rotated_obstacles = self._rotate_and_filter_obstacles(ego_pos, obstacles)
        # lidar_obs = []
        # for l in self.lidar_lines:
        #     min_distance = self.lidar_range
        #     for obs in rotated_obstacles:
        #         distance = l.intersection(obs).distance(ORIGIN)
        #         if distance>0.1 and distance<min_distance:
        #             min_distance = distance
        #     lidar_obs.append(min_distance)
        lidar_obs = self._fast_calc_lidar_obs(rotated_obstacles)
        return np.array(self.filter_obs(lidar_obs))
    
    def get_vehicle_base(self, ):
        lidar_base = []
        for l in self.lidar_lines:
            distance = l.intersection(VehicleBox).distance(ORIGIN)
            lidar_base.append(distance)
        return np.array(lidar_base)
    
    def filter_obs(self, lidar_obs):
        # for i in range(len(lidar_obs)):
        #     if abs(lidar_obs[i] - lidar_obs[i-1]) > 1.0 and\
        #     abs(lidar_obs[i] - lidar_obs[(i+1)%self.lidar_num]) > 1.0:
        #         lidar_obs[i] = (lidar_obs[i-1] + lidar_obs[(i+1)%self.lidar_num])/2
        return lidar_obs - self.vehicle_base # TODO: substract the length of vehicle

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




if __name__ == "__main__":
    from shapely.geometry import LinearRing
    import time
    import matplotlib.pyplot as plt

    obs1 = LinearRing(((1,1), (1,-1), (3,-1), (4,1), ))
    obs2 = LinearRing(((-5,1), (-5,-4), (-8,-1), (-9,1), ))
    obs3 = LinearRing(((0,4), (2, 6.3), (0.5, 7), (-3.3, 6.8)))

    raw_pos = [-10,5,0.9,0,0]
    car_pos = State(raw_pos)
    lidar_range = 10.0
    lidar_num = 120
    lidar = LidarSimlator(lidar_range, lidar_num)
    print(list(lidar.vehicle_base))
    OBSLIST = [obs1,obs2,obs3,obs1]

    t = time.time()
    lidar_view = lidar.get_observation(car_pos, OBSLIST)
    print(time.time()-t)

    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.set_xlim(-10,10)
    ax.set_ylim(-10,10)
    ax.set_xlabel('x')
    for i in range(len(lidar_view)):
        # line=plt.Line2D((0,0),(math.cos(i*math.pi/180)*lidar_view[i],math.sin(i*math.pi/180)*lidar_view[i]))
        ax.add_line(plt.Line2D((0,math.cos(i*math.pi/lidar_num*2)*lidar_view[i]), (0,math.sin(i*math.pi/lidar_num*2)*lidar_view[i])))
    for obs in lidar._rotate_and_filter_obstacles(raw_pos[:3], OBSLIST):
        ax.add_patch(plt.Polygon(xy=list(obs.coords), color='r'))
    ax.add_patch(plt.Polygon(xy=list(VehicleBox.coords), color='green'))
    plt.show()
