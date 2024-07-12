from typing import Callable, List
from enum import Enum
import copy

import numpy as np
from shapely.geometry import Point, LinearRing
from shapely.affinity import affine_transform

from configs import *



class Status(Enum):
    CONTINUE = 1
    ARRIVED = 2
    COLLIDED = 3
    OUTBOUND = 4
    OUTTIME = 5


class State:
    def __init__(self, raw_state: list):
        self.loc: Point = Point(raw_state[:2])
        self.heading: float = raw_state[2]
        if len(raw_state) == 3:
            self.speed: float = 0
            self.steering: float = 0
        else:
            self.speed: float = raw_state[3]
            self.steering: float = raw_state[4]

    def create_box(self) -> LinearRing:
        cos_theta = np.cos(self.heading)
        sin_theta = np.sin(self.heading)
        mat = [cos_theta, -sin_theta, sin_theta, cos_theta, self.loc.x, self.loc.y]
        return affine_transform(VehicleBox, mat)

    def get_pos(self,):
        return (self.loc.x, self.loc.y, self.heading)


class KSModel(object):
    """Update the state of a vehicle by the Kinematic Single-Track Model.

    Kinematic Single-Track Model use the vehicle's current speed, heading, location, 
    acceleration, and velocity of steering angle as input. Then it returns the estimation of 
    speed, heading, steering angle and location after a small time step.

    Use the center of vehicle's rear wheels as the origin of local coordinate system.

    Assume the vehicle is front-wheel-only drive.
    """
    def __init__(
        self, 
        wheel_base: float,
        step_len: float,
        n_step: int,
        speed_range: list,
        angle_range: list
    ):
        self.wheel_base = wheel_base
        self.step_len = step_len
        self.n_step = n_step
        self.speed_range = speed_range
        self.angle_range = angle_range
        self.mini_iter = 20

        # correct the turning radius
        self.correction = self.init_steer_correction()
    
    def init_steer_correction(self,):
        x,y,heading = 0,0,0
        ori_heading = heading
        steer,speed = self.angle_range[1], self.speed_range[1]
        iter = 10
        for _ in range(iter):
            for _ in range(self.mini_iter):
                x += speed * np.cos(heading) * self.step_len/self.mini_iter 
                y += speed * np.sin(heading) * self.step_len/self.mini_iter 
                heading += \
                    speed * np.tan(steer) / self.wheel_base * self.step_len/self.mini_iter 
        
        step_distance = speed*self.step_len*iter
        gt_radius = WHEEL_BASE/np.tan(VALID_STEER[-1])
        # gt_radius = WHEEL_BASE/np.tan(abs(steer))
        turn_radius_angle = step_distance/gt_radius
        x_origin = gt_radius*np.sin(ori_heading)
        y_origin = gt_radius*(1-np.cos(ori_heading))
        gt_x = gt_radius*np.sin(turn_radius_angle+ori_heading) - x_origin
        gt_y = gt_radius*(1-np.cos(turn_radius_angle+ori_heading)) - y_origin
        return gt_x/x, gt_y/y


    def step(self, state: State, action: list, step_time:int=NUM_STEP) -> State:
        """Update the state of a vehicle with the Kinematic Single-Track Model.

        Args:
            state (list): [x, y, car_angle, speed, steering]
            action (list): [steer, speed].
            step (float, optional): the step length for each simulation.
            n_step (int): number of step of updating the physical state. This value is decide by
                (physics simulation step length : rendering step length).

        """
        new_state = copy.deepcopy(state)
        x, y = new_state.loc.x, new_state.loc.y
        steer, speed = action
        new_state.steering = steer
        new_state.speed = speed
        new_state.speed = np.clip(new_state.speed, *self.speed_range)
        new_state.steering = np.clip(new_state.steering, *self.angle_range)


        # TODO: check correctness
        for _ in range(step_time):
            for _ in range(self.mini_iter):
                x += new_state.speed * np.cos(new_state.heading) * self.step_len/self.mini_iter
                y += new_state.speed * np.sin(new_state.heading) * self.step_len/self.mini_iter
                new_state.heading += \
                    new_state.speed * np.tan(new_state.steering) / self.wheel_base * self.step_len/self.mini_iter 

            # # correct the simulation error
            # kx, ky = self.correction
            # x0, y0, theta = state.loc.x, state.loc.y, state.heading
            # phi = np.arctan2(y-y0, x-x0)
            # alpha = phi - theta
            # r = np.sqrt((y-y0)**2 + (x-x0)**2)
            # xt_, yt_ = r*np.cos(alpha)*kx, r*np.sin(alpha)*ky
            # rt = np.sqrt(xt_**2 + yt_**2)
            # beta = np.arctan2(yt_, xt_)
            # x = rt*np.cos(theta+beta) + x0
            # y = rt*np.sin(theta+beta) + y0
        
        new_state.loc = Point(x, y)
        return new_state


class Vehicle:
    """_summary_
    """
    def __init__(
        self,
        wheel_base: float = WHEEL_BASE,
        step_len: float = STEP_LENGTH,
        n_step: int = NUM_STEP,
        speed_range: list = VALID_SPEED, 
        angle_range: list = VALID_STEER
    ) -> None:

        self.initial_state: list = None
        self.state: State = None
        self.box: LinearRing = None
        self.trajectory: List[State] = []
        self.kinetic_model: Callable = \
            KSModel(wheel_base, step_len, n_step, speed_range, angle_range)
        self.color = COLOR_POOL[0]
        self.v_max = None
        self.v_min = None

    def reset(self, initial_state: State):
        """
        Args:
            init_pos (list): [x0, y0, theta0]
        """
        self.initial_state = initial_state
        self.state = self.initial_state
        # self.color = random.sample(COLOR_POOL, 1)[0]
        self.v_max = self.initial_state.speed
        self.v_min = self.initial_state.speed
        self.box = self.state.create_box()
        self.trajectory.clear()
        self.trajectory.append(self.state)
        self.tmp_trajectory = self.trajectory.copy()

    def step(self, action: np.ndarray, step_time: int=NUM_STEP):
        """
        Args:
            action (list): [steer, speed]
        """
        prev_info = copy.deepcopy((self.state, self.box, self.v_max, self.v_min))
        self.state = self.kinetic_model.step(self.state, action, step_time)
        self.box = self.state.create_box()
        self.trajectory.append(self.state)
        self.tmp_trajectory.append(self.state)
        self.v_max = self.state.speed if self.state.speed > self.v_max else self.v_max
        self.v_min = self.state.speed if self.state.speed < self.v_min else self.v_min
        return prev_info

    def retreat(self, prev_info):
        '''
        Retreat the vehicle state from previous one.

        Args:
            prev_info (tuple): (state, box, v_max, v_min)
        '''
        self.state, self.box, self.v_max, self.v_min = prev_info
        self.trajectory.pop(-1)