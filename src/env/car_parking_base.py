'''
This is a collision-free env which only includes one parking case with random start position.
'''


import sys
sys.path.append("../")
from typing import Optional, Union
import csv
import math
from typing import OrderedDict, DefaultDict
import random

import numpy as np
import gym
from gym import spaces
from gym.error import DependencyNotInstalled, InvalidAction
from shapely.geometry import LineString, LinearRing, Polygon
from shapely.affinity import affine_transform
from heapdict import heapdict
try:
    # As pygame is necessary for using the environment (reset and step) even without a render mode
    #   therefore, pygame is a necessary import for the environment.
    import pygame
except ImportError:
    raise DependencyNotInstalled(
        "pygame is not installed, run `pip install pygame`"
    )

from env.vehicle import *
from env.map_base import *
from env.lidar_simulator import LidarSimlator
from env.parking_map_normal import ParkingMapNormal
from env.parking_map_dlp import ParkingMapDLP
import env.reeds_shepp as rsCurve
from env.observation_processor import Obs_Processor
from model.action_filter import ActionFilter2, ActionMask
from configs import *

DISCRETE_ACTION = np.array([
    [0, 0], # do nothing
    [-0.6, 0], # steer left
    [0.6, 0], # steer right
    [0, 0.2], # accelerate
    [0, -0.8], # decelerate
])

Valid_start_area = {
    '1': Polygon(((-19,-16), (0,-9), (-8,-3), (-19,-7), (-19,-16)))
}

class ParkingMap(object):
    default = {
        "path": '../data/Case%d.csv'
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

    def set_random_start(self, case_id):
        return
        if case_id==1: # TODO currently only for case 1
            random_start_area = Valid_start_area[str(case_id)]
            coords = np.array(list(random_start_area.boundary.coords))
            x_min, x_max = np.min(coords[:,0]), np.max(coords[:,0])
            y_min, y_max = np.min(coords[:,1]), np.max(coords[:,1])
            while True:
                random_x = random.random()*(x_max-x_min) + x_min
                random_y = random.random()*(y_max-y_min) + y_min
                random_pt = Point(random_x, random_y)
                if random_start_area.covers(random_pt):
                    # print('1')
                    random_angle = random.random()*math.pi*2
                    self.start = State([random_x, random_y, random_angle,0,0])
                    self.start_box = self.start.create_box()
                    # judge collision
                    collide = False
                    for obstacle in self.obstacles:
                        if self.start_box.intersects(obstacle.shape) or self.start_box.intersects(self.dest_box):
                            collide = True
                    if collide==False:
                        break
                # else:
                    # print('0')

    def reset(self, case_id: int = None, path: str = None) -> State:
        case_id =  np.random.randint(1, 21) if case_id is None else case_id
        path = self.default["path"] if path is None else path

        if case_id == self.case_id:
            self.set_random_start(case_id)
            return self.start
        else:
            self.case_id = case_id
            fname = path % case_id
            self.obstacles.clear()

        with open(fname, 'r') as f:
            reader = csv.reader(f)
            tmp = list(reader)
            v = [float(i) for i in tmp[0]]
            start_x, start_y = v[:2]
            # self.start = State(v[:3]+[0,0])
            self.start = State([v[0]-start_x, v[1]-start_y, v[2]])
            self.start_box = self.start.create_box()
            # self.dest = State(v[3:6]+[0,0])
            self.dest = State([v[3]-start_x, v[4]-start_y, v[5]])
            self.dest_box = self.dest.create_box()
            self.xmin = np.floor(min(self.start.loc.x, self.dest.loc.x) - 20)
            self.xmax = np.ceil(max(self.start.loc.x, self.dest.loc.x) + 20)
            self.ymin = np.floor(min(self.start.loc.y, self.dest.loc.y) - 20)
            self.ymax = np.ceil(max(self.start.loc.y, self.dest.loc.y) + 20)

            self.n_obstacle = int(v[6])
            n_vertex = np.array(v[7:7 + self.n_obstacle], dtype=np.int32)
            vertex_start = 7 + self.n_obstacle + (np.cumsum(n_vertex, dtype=np.int32) - n_vertex) * 2
            for vs, nv in zip(vertex_start, n_vertex):
                obst_coord = np.array(v[vs : vs+nv*2]).reshape((nv, 2), order='A')
                obst_coord[:,0] = obst_coord[:,0] - start_x
                obst_coord[:,1] = obst_coord[:,1] - start_y
                obstacle = LinearRing(obst_coord)
                self.obstacles.append(
                    Area(shape=obstacle, subtype="obstacle", color=(150, 150, 150, 255)))
            
        self.set_random_start(case_id)

        return self.start


class CarParking(gym.Env):
    """
    Description:

    Action Space:
        If continuous:
            There are 3 actions: steer (-1 is full left, +1 is full right), gas, and brake.
        If discrete:
            There are 5 actions: do nothing, steer left, steer right, gas, brake.

    Observation Space:
        State consists of 96x96 pixels.
    
    Args:
        
    """

    metadata = {
        "render_mode": [
            "human", 
            "rgb_array",
        ]
    }

    def __init__(
        self, 
        render_mode: str = None,
        fps: int = FPS,
        verbose: bool =True, 
        continuous: bool =True,
        use_lidar_observation: bool =USE_LIDAR,
        use_img_observation: bool=USE_IMG,
        use_action_mask: bool=USE_ACTION_MASK,
    ):
        super().__init__()

        self.verbose = verbose
        self.continuous = continuous
        self.use_lidar_observation = use_lidar_observation
        self.use_img_observation = use_img_observation
        self.use_action_mask = use_action_mask
        self.render_mode = "human" if render_mode is None else render_mode
        self.fps = fps
        self.screen: Optional[pygame.Surface] = None
        self.matrix = None
        self.clock = None
        self.is_open = True
        self.t = 0.0
        self.k = None
        self.level = MAP_LEVEL # ['Normal', 'Complex', 'Extrem']
        self.tgt_repr_size = 5 # relative_distance, cos(theta), sin(theta), cos(phi), sin(phi)

        if self.level in ['Normal', 'Complex', 'Extrem']:
            self.map = ParkingMapNormal(self.level)
        elif self.level == 'dlp':
            self.map = ParkingMapDLP()
        self.vehicle = Vehicle(n_step=NUM_STEP, step_len=STEP_LENGTH)
        self.lidar = LidarSimlator(LIDAR_RANGE, LIDAR_NUM)
        self.reward = 0.0
        self.prev_reward = 0.0
        self.accum_arrive_reward = 0.0
        self.collision_penalty = 0.0

        if self.continuous:
            # self.action_space = spaces.Box(
            #     np.array([VALID_ANGULAR_SPEED[0], VALID_ACCEL[0]]).astype(np.float32),
            #     np.array([VALID_ANGULAR_SPEED[1], VALID_ACCEL[1]]).astype(np.float32),
            # ) # steer, acceleration
            self.action_space = spaces.Box(
                np.array([VALID_STEER[0], VALID_SPEED[0]]).astype(np.float32),
                np.array([VALID_STEER[1], VALID_SPEED[1]]).astype(np.float32),
            ) # steer, speed
        else:
            self.action_space = spaces.Discrete(5) # do nothing, left, right, gas, brake
       
        self.observation_space = {}
        if self.use_action_mask:
            self.action_filter = ActionFilter2()
            # self.action_filter = ActionMask()
            self.observation_space['action_mask'] = spaces.Box(low=0, high=1, 
                shape=(N_DISCRETE_ACTION,), dtype=np.float64
            )
        if self.use_img_observation:
            self.img_processor = Obs_Processor()
            self.observation_space['img'] = spaces.Box(low=0, high=255, 
                shape=(OBS_W//self.img_processor.downsample_rate, OBS_H//self.img_processor.downsample_rate, 
                self.img_processor.n_channels), dtype=np.uint8
            )
            self.raw_img_shape = (OBS_W, OBS_H, 3)
        if self.use_lidar_observation:
            # the observation is composed of lidar points and target representation
            # the target representation is (relative_distance, cos(theta), sin(theta), cos(phi), sin(phi))
            # where the theta indicates the relative angle of parking lot, and phi means the heading of 
            # parking lot in the polar coordinate of the ego car's view
            low_bound, high_bound = np.zeros((LIDAR_NUM)), np.ones((LIDAR_NUM))*LIDAR_RANGE
            self.observation_space['lidar'] = spaces.Box(
                low=low_bound, high=high_bound, shape=(LIDAR_NUM,), dtype=np.float64
            )
        low_bound = np.array([0,-1,-1,-1,-1])
        high_bound = np.array([MAX_DIST_TO_DEST,1,1,1,1]) # TODO the hyper-param high_bound[0], the max distance to target
        self.observation_space['target'] = spaces.Box(
            low=low_bound, high=high_bound, shape=(self.tgt_repr_size,), dtype=np.float64
        )
    
    def set_level(self, level:str=None):
        if level is None:
            if random.random()>0.5:
                self.map = ParkingMapNormal()
            else:
                self.map = ParkingMap()
            return
        self.level = level
        if self.level in ['Normal', 'Complex', 'Extrem',]:
            self.map = ParkingMapNormal(self.level)
        elif self.level == 'dlp':
            self.map = ParkingMapDLP()

    def reset(self, case_id: int = None, data_dir: str = None, level: str = None, exchange_start_dest:bool = False) -> np.ndarray:
        self.reward = 0.0
        self.prev_reward = 0.0
        self.accum_arrive_reward = 0.0
        self.t = 0.0
        self.collision_penalty = 0.0

        if level is not None:
            self.set_level(level)
        if self.level == 'Mixed':
            self.set_level()
        initial_state = self.map.reset(case_id, data_dir)
        if exchange_start_dest:
            self.map.dest, self.map.start = self.map.start, self.map.dest
            self.map.dest_box, self.map.start_box = self.map.start_box, self.map.dest_box
            initial_state = self.map.start
        self.vehicle.reset(initial_state)
        self.matrix = self.coord_transform_matrix()
        return self.step()[0]

    def coord_transform_matrix(self) -> list:
        """Get the transform matrix that convert the real world coordinate to the pygame coordinate.
        """
        # TODO the ratio k
        # k1 = WIN_W/ (self.map.xmax - self.map.xmin)
        # k2 = WIN_H / (self.map.ymax - self.map.ymin)
        k = K
        bx = 0.5 * (WIN_W - k * (self.map.xmax + self.map.xmin))
        by = 0.5 * (WIN_H - k * (self.map.ymax + self.map.ymin))
        self.k = k
        return [k, 0, 0, k, bx, by]

    def _coord_transform(self, object) -> list:
        transformed = affine_transform(object, self.matrix)
        return list(transformed.coords)

    def _detect_collision(self):
        # return False
        for obstacle in self.map.obstacles:
            if self.vehicle.box.intersects(obstacle.shape):
                return True
        return False
    
    def _detect_outbound(self):
        x, y = self.vehicle.state.loc.x, self.vehicle.state.loc.y
        return x>self.map.xmax or x<self.map.xmin or y>self.map.ymax or y<self.map.ymin
        vehicle_box = np.array(self._coord_transform(self.vehicle.box))
        if vehicle_box[:, 0].min() < 0:
            return True
        if vehicle_box[:, 0].max() > WIN_W:
            return True
        if vehicle_box[:, 1].min() < 0:
            return True
        if vehicle_box[:, 1].max() > WIN_H:
            return True
        return False

    def _check_arrived(self):
        vehicle_box = Polygon(self.vehicle.box)
        dest_box = Polygon(self.map.dest_box)
        union_area = vehicle_box.intersection(dest_box).area
        if union_area / dest_box.area > 0.95:
            # print('arrive!!: ',union_area / dest_box.area)
            return True
        return False
    
    def _check_time_exceeded(self):
        return self.t > TOLERANT_TIME

    def _check_status(self):
        # if self._detect_collision():
        #     return Status.COLLIDED
        if self._detect_outbound():
            return Status.OUTBOUND
        if self._check_arrived():
            return Status.ARRIVED
        if self._check_time_exceeded():
            return Status.OUTTIME
        return Status.CONTINUE

    def _get_reward(self, prev_state: State, curr_state: State):

        # time penalty
        time_cost = - np.tanh(self.t / (10*TOLERANT_TIME))

        # RS distance reward
        if REWARD_WEIGHT['rs_dist_reward'] != 0:
            radius = math.tan(VALID_STEER[-1])/WHEEL_BASE
            curr_rs_dist = rsCurve.calc_optimal_path(*curr_state.get_pos(), *self.map.dest.get_pos(), radius , 0.1).L
            prev_rs_dist = rsCurve.calc_optimal_path(*prev_state.get_pos(), *self.map.dest.get_pos(), radius, 0.1).L
            rs_dist_norm_ratio = rsCurve.calc_optimal_path(*self.map.start.get_pos(), *self.map.dest.get_pos(), radius, 0.1).L
            rs_dist_reward = math.exp(-curr_rs_dist/rs_dist_norm_ratio) - \
                math.exp(-prev_rs_dist/rs_dist_norm_ratio)
        else:
            rs_dist_reward = 0

        # Euclidean distance reward & angle reward
        def get_angle_diff(angle1, angle2):
            # norm to 0 ~ pi/2
            angle_dif = math.acos(math.cos(angle1 - angle2)) # 0~pi
            return angle_dif if angle_dif<math.pi/2 else math.pi-angle_dif
        dist_diff = curr_state.loc.distance(self.map.dest.loc)
        angle_diff = get_angle_diff(curr_state.heading, self.map.dest.heading)
        prev_dist_diff = prev_state.loc.distance(self.map.dest.loc)
        prev_angle_diff = get_angle_diff(prev_state.heading, self.map.dest.heading)
        dist_norm_ratio = max(self.map.dest.loc.distance(self.map.start.loc),10)
        angle_norm_ratio = math.pi
        # dist_reward = math.exp(-dist_diff/dist_norm_ratio) - \
            # math.exp(-prev_dist_diff/dist_norm_ratio)
        dist_reward = prev_dist_diff/dist_norm_ratio - dist_diff/dist_norm_ratio
        # angle_reward = math.exp(-angle_diff/angle_norm_ratio) - \
        #     math.exp(-prev_angle_diff/angle_norm_ratio)
        angle_reward = prev_angle_diff/angle_norm_ratio - angle_diff/angle_norm_ratio
        
        # Box union reward
        vehicle_box = Polygon(self.vehicle.box)
        dest_box = Polygon(self.map.dest_box)
        union_area = vehicle_box.intersection(dest_box).area
        box_union_reward = union_area/(2*dest_box.area - union_area)
        if box_union_reward < self.accum_arrive_reward:
            box_union_reward = 0 
        else:
            prev_arrive_reward = self.accum_arrive_reward
            self.accum_arrive_reward = box_union_reward
            box_union_reward -= prev_arrive_reward
        return [time_cost ,rs_dist_reward ,dist_reward ,angle_reward ,box_union_reward]
        
    def get_reward(self, status, prev_state):
        reward_info = [0,0,0,0,0]
        if status == Status.CONTINUE:
            reward_info = self._get_reward(prev_state, self.vehicle.state)
        return reward_info

    def step(self, action: Union[np.ndarray, int] = None):
        '''
        Parameters:
        ----------
        `action`: `np.ndarra`y if continous action space else `int`

        Returns:
        ----------
        ``obsercation`` (Dict): 
            the observation of image based surroundings, lidar view and target representation.
            If `use_lidar_observation` is `True`, then `obsercation['img'] = None`.
            If `use_lidar_observation` is `False`, then `obsercation['lidar'] = None`. 

        ``reward_info`` (OrderedDict): different types of reward information, including:
                time_cost ,rs_dist_reward ,dist_reward ,angle_reward ,box_union_reward
        `status` (`Status`): represent the state of vehicle, including:
                `CONTINUE`, `ARRIVED`, `COLLIDED`, `OUTBOUND`, `OUTTIME`
        `info` (`OrderedDict`): other information.
        '''
        assert self.vehicle is not None
        prev_state = self.vehicle.state
        collide = False
        arrive = False
        if action is not None: # 5ms
            for simu_step_num in range(NUM_STEP):
                if self.continuous:
                    prev_info = self.vehicle.step(action,step_time=1)
                else:
                    if not self.action_space.contains(action):
                        raise InvalidAction(
                            f"you passed the invalid action `{action}`. "
                            f"The supported action_space is `{self.action_space}`"
                        )
                    prev_info = self.vehicle.step(DISCRETE_ACTION[action],step_time=1)
                if self._check_arrived():
                    arrive = True
                    break
                if self._detect_collision():
                    if simu_step_num == 0:
                        collide = ENV_COLLIDE
                        self.vehicle.retreat(prev_info)
                    else:
                        self.vehicle.retreat(prev_info)
                    simu_step_num -= 1
                    break
            simu_step_num += 1
            # remove redundant trajectory
            if simu_step_num > 1:
                del self.vehicle.trajectory[-simu_step_num:-1]

        self.t += 1 # . / self.fps
        observation = self.render(self.render_mode) # 11 ms
        if arrive:
            status = Status.ARRIVED
        else:
            status = Status.COLLIDED if collide else self._check_status()

        reward_list = self.get_reward(status, prev_state) # <1 ms
        # time_cost ,rs_dist_reward ,dist_reward ,angle_reward ,box_union_reward
        reward_info = OrderedDict({'time_cost':reward_list[0],\
            'rs_dist_reward':reward_list[1],\
            'dist_reward':reward_list[2],\
            'angle_reward':reward_list[3],\
            'box_union_reward':reward_list[4],})

        # done = False if status==Status.CONTINUE else True
        info = OrderedDict({'reward_info':reward_info,
            'path_to_dest':None})
        if self.t > 1 and status==Status.CONTINUE and random.random()<P_RS\
            and self.vehicle.state.loc.distance(self.map.dest.loc)<RS_MAX_DIST: # 15 ms
            # t = time.time()
            rs_path_to_dest = self.find_rs_path(status)
            # print('rs ', time.time()-t)
            if rs_path_to_dest is not None:
                info['path_to_dest'] = rs_path_to_dest
                # print(self.vehicle.state.loc)
                # for i in range(len(rs_path_to_dest)):
                #     print(rs_path_to_dest[i])
                #     if i%2 == 0 or i==len(rs_path_to_dest)-1:
                #         self.vehicle.state = State(rs_path_to_dest[i]+[0,0])
                #         self.vehicle.box = self.vehicle.state.create_box()
                #         self.vehicle.trajectory.append(self.vehicle.state.loc)
                #         self.render(self.render_mode)

        return observation, reward_info, status, info

    def _render(self, surface: pygame.Surface):
        surface.fill(BG_COLOR)
        for obstacle in self.map.obstacles:
            pygame.draw.polygon(
                surface, OBSTACLE_COLOR, self._coord_transform(obstacle.shape))

        pygame.draw.polygon( #
            surface, START_COLOR, self._coord_transform(self.map.start_box), width=1)
        pygame.draw.polygon(
            surface, DEST_COLOR, self._coord_transform(self.map.dest_box))#, width=1
        
        pygame.draw.polygon(
            surface, self.vehicle.color, self._coord_transform(self.vehicle.box))

        if RENDER_TRAJ and len(self.vehicle.trajectory) > 1:
            render_len = min(len(self.vehicle.trajectory), TRAJ_RENDER_LEN)
            # for i in range(len(self.vehicle.trajectory) - render_len):
            #     vehicle_box = self.vehicle.trajectory[i].create_box()
            #     pygame.draw.polygon(
            #         surface, TRAJ_COLORS[0], self._coord_transform(vehicle_box))
            for i in range(render_len):
                vehicle_box = self.vehicle.trajectory[-(render_len-i)].create_box()
                pygame.draw.polygon(
                    surface, TRAJ_COLORS[-(render_len-i)], self._coord_transform(vehicle_box))

        # if len(self.vehicle.trajectory) > 1:
        #     traj = [t.loc for t in self.vehicle.trajectory]
        #     pygame.draw.lines(
        #         surface, TRAJ_COLOR, False, 
        #         self._coord_transform(LineString(traj))
        #     )

    def _get_img_observation(self, surface: pygame.Surface):
        angle = self.vehicle.state.heading
        old_center = surface.get_rect().center

        # Rotate and find the center of the vehicle
        capture = pygame.transform.rotate(surface, np.rad2deg(angle))
        rotate = pygame.Surface((WIN_W, WIN_H))
        rotate.blit(capture, capture.get_rect(center=old_center))
        
        vehicle_center = np.array(self._coord_transform(self.vehicle.box.centroid)[0])
        dx = (vehicle_center[0]-old_center[0])*np.cos(angle) \
            + (vehicle_center[1]-old_center[1])*np.sin(angle)
        dy = -(vehicle_center[0]-old_center[0])*np.sin(angle) \
            + (vehicle_center[1]-old_center[1])*np.cos(angle)
        
        # align the center of the observation with the center of the vehicle
        observation = pygame.Surface((WIN_W, WIN_H))
    
        observation.fill(BG_COLOR)
        observation.blit(rotate, (int(-dx), int(-dy)))
        observation = observation.subsurface((
            (WIN_W-OBS_W)/2, (WIN_H-OBS_H)/2), (OBS_W, OBS_H))

        # add destination info when it is out of horizon
        if DRAW_DEST_DIRECTION and self.map.dest_box.centroid.distance(self.vehicle.box.centroid)>150/self.k:
            dest_direction_relative = math.atan2(self.map.dest_box.centroid.y-self.vehicle.box.centroid.y, \
                self.map.dest_box.centroid.x-self.vehicle.box.centroid.x) - self.vehicle.state.heading
            arrow_margin = 20
            if -OBS_H/OBS_W < math.tan(dest_direction_relative) <= OBS_H/OBS_W:
                arrow_pos_x = OBS_W - arrow_margin if math.cos(dest_direction_relative)>0 else arrow_margin
                arrow_pos_y = (arrow_pos_x-OBS_W/2) * math.tan(dest_direction_relative) + OBS_H/2
            else:
                arrow_pos_y = OBS_H - arrow_margin if math.sin(dest_direction_relative)>0 else arrow_margin
                arrow_pos_x = (arrow_pos_y-OBS_H/2) * math.cos(dest_direction_relative) / math.sin(dest_direction_relative) + OBS_W/2
            # create the arrow
            arrow_margin -= 2
            pt1 = (int(arrow_pos_x+math.cos(dest_direction_relative)*arrow_margin),\
                    int(arrow_pos_y+math.sin(dest_direction_relative)*arrow_margin))
            pt2 = (int(arrow_pos_x+math.cos(dest_direction_relative+1.7)*arrow_margin/2),\
                    int(arrow_pos_y+math.sin(dest_direction_relative+1.7)*arrow_margin/2))
            pt3 = (int(arrow_pos_x), int(arrow_pos_y))
            pt4 = (int(arrow_pos_x+math.cos(dest_direction_relative-1.7)*arrow_margin/2),\
                    int(arrow_pos_y+math.sin(dest_direction_relative-1.7)*arrow_margin/2))
            pygame.draw.polygon(observation, DEST_COLOR, [pt1, pt2, pt3, pt4])

    
        obs_str = pygame.image.tostring(observation, "RGB")
        observation = np.frombuffer(obs_str, dtype=np.uint8)
        observation = observation.reshape(self.raw_img_shape)

        return observation

    def _process_img_observation(self, img):
        '''
        Process the img into channels of different information.

        Parameters
        ------
        img (np.ndarray): RGB image of shape (OBS_W, OBS_H, 3)

        Returns
        ------
        processed img (np.ndarray): shape (OBS_W//downsample_rate, OBS_H//downsample_rate, n_channels )
        '''
        processed_img = self.img_processor.process_img(img)
        return processed_img

    def _get_lidar_observation(self,):
        obs_list = [obs.shape for obs in self.map.obstacles]
        lidar_view = self.lidar.get_observation(self.vehicle.state, obs_list)
        return lidar_view
    
    def _get_targt_repr(self,):
        # target position representation
        dest_pos = (self.map.dest.loc.x, self.map.dest.loc.y, self.map.dest.heading)
        ego_pos = (self.vehicle.state.loc.x, self.vehicle.state.loc.y, self.vehicle.state.heading)
        rel_distance = math.sqrt((dest_pos[0]-ego_pos[0])**2 + (dest_pos[1]-ego_pos[1])**2)
        rel_angle = math.atan2(dest_pos[1]-ego_pos[1], dest_pos[0]-ego_pos[0]) - ego_pos[2]
        rel_dest_heading = dest_pos[2] - ego_pos[2]
        tgt_repr = np.array([rel_distance, math.cos(rel_angle), math.sin(rel_angle),\
            math.cos(rel_dest_heading), math.cos(rel_dest_heading)])
        return tgt_repr 

    def render(self, mode: str = "human"):
        assert mode in self.metadata["render_mode"]
        assert self.vehicle is not None

        if mode == "human":
            display_flags = pygame.SHOWN
        else:
            display_flags = pygame.HIDDEN
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((WIN_W, WIN_H), flags = display_flags)
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self._render(self.screen)
        observation = {'img':None, 'lidar':None, 'target':None, 'action_mask':None}
        if self.use_img_observation:
            raw_observation = self._get_img_observation(self.screen)
            observation['img'] = self._process_img_observation(raw_observation)
        if self.use_lidar_observation:
            observation['lidar'] = self._get_lidar_observation() # 2 ms
        if self.use_action_mask:
            # t = time.time()
            observation['action_mask'] = self.action_filter.get_steps(observation['lidar']) # 7 ms
            # t1 = time.time() - t
            # t = time.time()
            # obs_tmp = self.action_mask.get_steps(observation['lidar'])
            # t2 = time.time() - t
            # print('time: ', t1, t2)
            # if np.sum(observation['action_mask']-obs_tmp) != 0:
            #     print(observation['action_mask'])
            #     print(obs_tmp)
            #     print(observation['action_mask']-obs_tmp)
            #     print("-"*10)
            #     import matplotlib.pyplot as plt
            #     plt.subplot(1,2,1)
            #     plt.imshow(observation['img'])
            #     plt.subplot(1,2,2)
            #     angle = np.pi*2*np.arange(len(observation['lidar']))/len(observation['lidar'])
            #     xs = np.cos(angle)*(observation['lidar']+vehicle_base)
            #     ys = np.sin(angle)*(observation['lidar']+vehicle_base)
            #     plt.plot(xs, ys)
            #     plt.plot(VehicleBox.coords.xy[0], VehicleBox.coords.xy[1])
            #     plt.show()
            #     p
        observation['target'] = self._get_targt_repr()
        pygame.display.update()
        self.clock.tick(self.fps)
        
        return observation

    def find_rs_path(self,status):
        '''
        Find collision-free RS path. 

        Returns:
            path (PATH): the related PATH object which is collision-free.
        '''
        startX, startY, startYaw = self.vehicle.state.loc.x, self.vehicle.state.loc.y, self.vehicle.state.heading
        goalX, goalY, goalYaw = self.map.dest.loc.x, self.map.dest.loc.y, self.map.dest.heading
        radius = math.tan(VALID_STEER[-1])/WHEEL_BASE
        #  Find all possible reeds-shepp paths between current and goal node
        reedsSheppPaths = rsCurve.calc_all_paths(startX, startY, startYaw, goalX, goalY, goalYaw, radius, 0.1)

        # Check if reedsSheppPaths is empty
        if not reedsSheppPaths:
            return None

        # Find path with lowest cost considering non-holonomic constraints
        costQueue = heapdict()
        for path in reedsSheppPaths:
            costQueue[path] = path.L

        # Find first path in priority queue that is collision free
        min_path_len = -1
        idx = 0
        while len(costQueue)!=0:
            idx += 1
            path = costQueue.popitem()[0]
            if min_path_len < 0:
                min_path_len = path.L
            # if path.L > 1.6*min_path_len and idx > 2:
            if path.L > 2*min_path_len:
                # print('BREAK!!', path.L, min_path_len)
                break
            traj=[]
            traj = [[path.x[k],path.y[k],path.yaw[k]] for k in range(len(path.x))]
            # TODO: check the trajectory is too long
            # traj_vlid1 = self.is_traj_valid2(traj)
            traj_valid2 = self.is_traj_valid(traj)
            # print(traj_vlid1,  traj_valid2)
            # if traj_valid2 != traj_vlid1:
            #     p
            if traj_valid2:
                return path
        return None

    def is_traj_valid2(self, traj):
        '''
        Check whether the trajectory is collision-free and in range.
        '''
        for t in traj:
            x,y = t[0], t[1]
            if x < self.map.xmin or x > self.map.xmax \
                or y < self.map.ymin or y > self.map.ymax:
                return False
            cos_theta = np.cos(t[2])
            sin_theta = np.sin(t[2])
            mat = [cos_theta, -sin_theta, sin_theta, cos_theta, t[0], t[1]]
            vehicle_box = affine_transform(VehicleBox, mat)
            for obstacle in self.map.obstacles:
                if vehicle_box.intersects(obstacle.shape):
                    return False
        return True
    
    def is_traj_valid(self, traj):
        # t1 = time.time()
        # vx1s, vx2s, vy1s, vy2s = [], [], [], []
        # for t in traj:
        #     x,y = t[0], t[1]
        #     if x < self.map.xmin or x > self.map.xmax \
        #         or y < self.map.ymin or y > self.map.ymax:
        #         return False
        #     cos_theta = np.cos(t[2])
        #     sin_theta = np.sin(t[2])
        #     mat = [cos_theta, -sin_theta, sin_theta, cos_theta, t[0], t[1]]
        #     vehicle_box = affine_transform(VehicleBox, mat)

        #     vehicle_coords = np.array(vehicle_box.coords) # (5,2)
        #     vx1s.append(vehicle_coords[:-1, 0].reshape(-1,1)) # (4,1)
        #     vx2s.append(vehicle_coords[1:, 0].reshape(-1,1))
        #     vy1s.append(vehicle_coords[:-1, 1].reshape(-1,1))
        #     vy2s.append(vehicle_coords[1:, 1].reshape(-1,1))

        # vx1s = np.concatenate(vx1s, axis=0)
        # vx2s = np.concatenate(vx2s, axis=0)
        # vy1s = np.concatenate(vy1s, axis=0)
        # vy2s = np.concatenate(vy2s, axis=0)
        car_coords1 = np.array(VehicleBox.coords)[:4] # (4,2)
        car_coords2 = np.array(VehicleBox.coords)[1:] # (4,2)
        car_coords_x1 = car_coords1[:,0].reshape(1,-1)
        car_coords_y1 = car_coords1[:,1].reshape(1,-1) # (1,4)
        car_coords_x2 = car_coords2[:,0].reshape(1,-1)
        car_coords_y2 = car_coords2[:,1].reshape(1,-1) # (1,4)
        vxs = np.array([t[0] for t in traj])
        vys = np.array([t[1] for t in traj])
        # check outbound
        if np.min(vxs) < self.map.xmin or np.max(vxs) > self.map.xmax \
        or np.min(vys) < self.map.ymin or np.max(vys) > self.map.ymax:
            return False
        vthetas = np.array([t[2] for t in traj])
        cos_theta = np.cos(vthetas).reshape(-1,1) # (T,1)
        sin_theta = np.sin(vthetas).reshape(-1,1)
        vehicle_coords_x1 = cos_theta*car_coords_x1 - sin_theta*car_coords_y1 + vxs.reshape(-1,1) # (T,4)
        vehicle_coords_y1 = sin_theta*car_coords_x1 + cos_theta*car_coords_y1 + vys.reshape(-1,1)
        vehicle_coords_x2 = cos_theta*car_coords_x2 - sin_theta*car_coords_y2 + vxs.reshape(-1,1) # (T,4)
        vehicle_coords_y2 = sin_theta*car_coords_x2 + cos_theta*car_coords_y2 + vys.reshape(-1,1)
        vx1s = vehicle_coords_x1.reshape(-1,1)
        vx2s = vehicle_coords_x2.reshape(-1,1)
        vy1s = vehicle_coords_y1.reshape(-1,1)
        vy2s = vehicle_coords_y2.reshape(-1,1)
        # Line 1: the edges of vehicle box, ax + by + c = 0
        a = (vy2s - vy1s).reshape(-1,1) # (4*t,1)
        b = (vx1s - vx2s).reshape(-1,1)
        c = (vy1s*vx2s - vx1s*vy2s).reshape(-1,1)
        # print('prepare vehicle', time.time()-t1)
        
        # convert obstacles(LinerRing) to edges ((x1,y1), (x2,y2))
        # t1 = time.time()
        x_max = np.max(vx1s) + 5
        x_min = np.min(vx1s) - 5
        y_max = np.max(vy1s) + 5
        y_min = np.min(vy1s) - 5
        # x1s, x2s, y1s, y2s = self.map_obstacle_edge # (4, E, 4)
        # if len(x1s) == 0: # no obstacle around
        #     return False
        # # print((np.sum((y2s < y_min), axis=-1) == 4))
        # consider_obst_idx = \
        # (np.sum((x1s > x_max), axis=-1) == 4) + (np.sum((x1s < x_min), axis=-1) == 4) +\
        #     (np.sum((y1s > y_max), axis=-1) == 4) + (np.sum((y2s < y_min), axis=-1) == 4)
        # consider_obst_idx = (consider_obst_idx==False)
        # # print(consider_obst_idx)
        # # print(len(consider_obst_idx))
        # x1s = x1s[consider_obst_idx,:]
        # x2s = x2s[consider_obst_idx,:]
        # y1s = y1s[consider_obst_idx,:]
        # y2s = y2s[consider_obst_idx,:]
        # # print(len(self.map.obstacles), len(x1s))
        # # print('preprare obst 1, ', time.time()-t1)

        x1s, x2s, y1s, y2s = [], [], [], []
        for obst in self.map.obstacles:
            if isinstance(obst, Area):
                obst = obst.shape
            obst_coords = np.array(obst.coords) # (n+1,2)
            if (obst_coords[:,0] > x_max).all() or (obst_coords[:,0] < x_min).all()\
                or (obst_coords[:,1] > y_max).all() or (obst_coords[:,1] < y_min).all():
                continue
            x1s.extend(list(obst_coords[:-1, 0]))
            x2s.extend(list(obst_coords[1:, 0]))
            y1s.extend(list(obst_coords[:-1, 1]))
            y2s.extend(list(obst_coords[1:, 1]))
        # print('preprare obst 1, ', time.time()-t1)
        if len(x1s) == 0: # no obstacle around
            return True
        x1s, x2s, y1s, y2s  = np.array(x1s).reshape(1,-1), np.array(x2s).reshape(1,-1),\
            np.array(y1s).reshape(1,-1), np.array(y2s).reshape(1,-1), 
        # Line 2: the edges of obstacles, dx + ey + f = 0
        d = (y2s - y1s).reshape(1,-1) # (1,E)
        e = (x1s - x2s).reshape(1,-1)
        f = (y1s*x2s - x1s*y2s).reshape(1,-1)
        # print('preprare obstacle', time.time()-t1)
        # t1 = time.time()

        # calculate the intersections
        det = a*e - b*d # (4, E)
        parallel_line_pos = (det==0) # (4, E)
        det[parallel_line_pos] = 1 # temporarily set "1" to avoid "divided by zero"
        raw_x = (b*f - c*e)/det # (4, E)
        raw_y = (c*d - a*f)/det
        # print('prepare: ',time.time()-t1, len(vx1s), len(x1s[0]))

        collide_map_x = np.ones_like(raw_x, dtype=np.uint8)
        collide_map_y = np.ones_like(raw_x, dtype=np.uint8)
        # the false positive intersections on line L2(not on edge L2)
        collide_map_x[raw_x>np.maximum(x1s, x2s)] = 0
        collide_map_x[raw_x<np.minimum(x1s, x2s)] = 0
        collide_map_y[raw_y>np.maximum(y1s, y2s)] = 0
        collide_map_y[raw_y<np.minimum(y1s, y2s)] = 0
        # the false positive intersections on line L1(not on edge L1)
        collide_map_x[raw_x>np.maximum(vx1s, vx2s)] = 0
        collide_map_x[raw_x<np.minimum(vx1s, vx2s)] = 0
        collide_map_y[raw_y>np.maximum(vy1s, vy2s)] = 0
        collide_map_y[raw_y<np.minimum(vy1s, vy2s)] = 0

        collide_map = collide_map_x*collide_map_y
        collide_map[parallel_line_pos] = 0
        collide = np.sum(collide_map) > 0
        # print('traj valid: ', time.time()-t1)

        if collide:
            return False
        return True

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            self.is_open = False
            pygame.quit()


if __name__ == "__main__":
    a = np.array([0.0, 0.0])

    def register_input():
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    a[0] = -0.1
                if event.key == pygame.K_RIGHT:
                    a[0] = +0.1
                if event.key == pygame.K_UP:
                    a[1] = -0.1
                if event.key == pygame.K_DOWN:
                    a[1] = -0.1  # set 1.0 for wheels to block to zero rotation
                if event.key == pygame.K_RETURN:
                    global restart
                    restart = True

    env = CarParking()
    is_open = True
    while is_open:
        env.reset()
        total_reward = 0.0
        n_step = 0
        restart = False
        status = Status.CONTINUE
        while status == Status.CONTINUE and is_open:
            register_input()
            observation, reward, status = env.step(a)
            total_reward += reward
            n_step += 1

        is_open = False

    env.close()

