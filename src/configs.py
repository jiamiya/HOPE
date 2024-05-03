
import os
os.environ["SDL_VIDEODRIVER"]="dummy"
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

import numpy as np
import torch
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
SEED = 42

#########################
# vehicle
# WHEEL_BASE = 2.8  # wheelbase
# FRONT_HANG = 0.96  # front hang length
# REAR_HANG = 0.93  # rear hang length
# LENGTH = WHEEL_BASE+FRONT_HANG+REAR_HANG
# WIDTH = 1.94  # width

WHEEL_BASE = 2.65  # wheelbase
FRONT_HANG = 0.905  # front hang length
REAR_HANG = 0.96  # rear hang length
LENGTH = 4.515
WIDTH = 1.86  # width


from shapely.geometry import LinearRing
VehicleBox = LinearRing([
    (-REAR_HANG, -WIDTH/2), 
    (FRONT_HANG + WHEEL_BASE, -WIDTH/2), 
    (FRONT_HANG + WHEEL_BASE,  WIDTH/2),
    (-REAR_HANG,  WIDTH/2)])

COLOR_POOL = [
    (30, 144, 255, 255), # dodger blue
    (255, 127, 80, 255), # coral
    (255, 215, 0, 255) # gold
]

VALID_SPEED = [-2.5, 2.5]
# VALID_STEER = [-0.75, 0.75]
VALID_STEER = [-0.449, 0.449]
VALID_ACCEL = [-1.0, 1.0]
VALID_ANGULAR_SPEED = [-0.5, 0.5]

NUM_STEP = 10
STEP_LENGTH = 5e-2

########################
# senerio
MAP_LEVEL = 'Normal' # ['Normal', 'Complex', 'Extrem', 'dlp']
MIN_PARK_LOT_LEN_DICT = {'Extrem':LENGTH+0.6,
                            'Complex':LENGTH+0.9,
                            'Normal':LENGTH*1.25,}
MAX_PARK_LOT_LEN_DICT = {'Extrem':LENGTH+0.9,
                            'Complex':LENGTH*1.25,
                            'Normal':LENGTH*1.25+0.5}
MIN_PARK_LOT_WIDTH_DICT = {
    'Complex':WIDTH+0.4,
    'Normal':WIDTH+0.85,
}
MAX_PARK_LOT_WIDTH_DICT = {
    'Complex':WIDTH+0.85,
    'Normal':WIDTH+1.2,
}
PARA_PARK_WALL_DIST_DICT = {
    'Extrem':3.5,
    'Complex':4.0,
    'Normal':4.5,
}
BAY_PARK_WALL_DIST_DICT = {
    # 'Extrem':5.0,
    'Complex':6.0,
    'Normal':7.0,
}
N_OBSTACLE_DICT = {
    'Extrem':8,
    'Complex':5,
    'Normal':3,
}

# Normal level
MIN_DIST_TO_OBST = 0.1
MAX_DRIVE_DISTANCE = 15.0
DROUP_OUT_OBST = 0.0

#########################
# env
ENV_COLLIDE = False
BG_COLOR = (255, 255, 255, 255)
START_COLOR = (100, 149, 237, 255)
DEST_COLOR = (69, 139, 0, 255)
OBSTACLE_COLOR = (150, 150, 150, 255)
TRAJ_COLOR_HIGH = (10, 10, 200, 255)
TRAJ_COLOR_LOW = (10, 10, 10, 255)
TRAJ_RENDER_LEN = 20
TRAJ_COLORS = list(map(tuple,np.linspace(\
    np.array(TRAJ_COLOR_LOW), np.array(TRAJ_COLOR_HIGH), TRAJ_RENDER_LEN, endpoint=True, dtype=np.uint8)))
OBS_W = 256
OBS_H = 256
VIDEO_W = 600
VIDEO_H = 400
WIN_W = 500
WIN_H = 500
LIDAR_RANGE = 10.0
LIDAR_NUM = 120
FPS = 100
TOLERANT_TIME = 200
USE_LIDAR = True
USE_IMG = True
USE_ACTION_MASK = True
MAX_DIST_TO_DEST = 20
K = 12 # the render scale
RS_MAX_DIST = 10
RENDER_TRAJ = True


PRECISION = 10
step_speed = 1
discrete_actions = []
for i in np.arange(VALID_STEER[-1], -(VALID_STEER[-1] + VALID_STEER[-1]/PRECISION), -VALID_STEER[-1]/PRECISION):
    discrete_actions.append([i, step_speed])
for i in np.arange(VALID_STEER[-1], -(VALID_STEER[-1] + VALID_STEER[-1]/PRECISION), -VALID_STEER[-1]/PRECISION):
    discrete_actions.append([i, -step_speed])
N_DISCRETE_ACTION = len(discrete_actions)

#########################
# model
GAMMA = 0.98
BATCH_SIZE = 8192
LR = 5e-6
TAU = 0.1
MAX_TRAIN_STEP = 1e6
ORTHOGONAL_INIT = True
LR_DECAY = False
UPDATE_IMG_ENCODE = False


C_CONV = [4, 8,]
SIZE_FC = [256]

ATTENTION_CONFIG = {
                'depth': 1,
                'heads': 8,
                'dim_head': 32,
                'mlp_dim': 128,
                'hidden_dim': 128,
    }
USE_ATTENTION = True

ACTOR_CONFIGS = {
    'n_modal':2+int(USE_IMG)+int(USE_ACTION_MASK),
    'lidar_shape':LIDAR_NUM,
    'target_shape':5,
    'action_mask_shape':N_DISCRETE_ACTION if USE_ACTION_MASK else None,
    'img_shape':(3,64,64) if USE_IMG else None,
    'output_size':2,
    'embed_size':128,
    'hidden_size':256,
    'n_hidden_layers':3,
    'n_embed_layers':2,
    'img_conv_layers':C_CONV,
    'img_linear_layers':SIZE_FC,
    'k_img_conv':3,
    'orthogonal_init':True,
    'use_tanh_output':True,
    'use_tanh_activate':True,
    'attention_configs': ATTENTION_CONFIG if USE_ATTENTION else None,
}

CRITIC_CONFIGS = {
    'n_modal':2+int(USE_IMG)+int(USE_ACTION_MASK),
    'lidar_shape':LIDAR_NUM,
    'target_shape':5,
    'action_mask_shape':N_DISCRETE_ACTION if USE_ACTION_MASK else None,
    'img_shape':(3,64,64) if USE_IMG else None,
    'output_size':1,
    'embed_size':128,
    'hidden_size':256,
    'n_hidden_layers':3,
    'n_embed_layers':2,
    'img_conv_layers':C_CONV,
    'img_linear_layers':SIZE_FC,
    'k_img_conv':3,
    'orthogonal_init':True,
    'use_tanh_output':False,
    'use_tanh_activate':True,
    'attention_configs': ATTENTION_CONFIG if USE_ATTENTION else None,
}

REWARD_RATIO = 0.1
from typing import OrderedDict
REWARD_WEIGHT = OrderedDict({'time_cost':1,\
            'rs_dist_reward':0,\
            'dist_reward':5,\
            'angle_reward':0,\
            'box_union_reward':10,})

# vehicle_boundary = np.load('../data/vehicle_boundary.npy')

CONFIGS_ACTION = {
    'use_tanh_activate': True,
    'hidden_size':256,
    'lidar_shape':LIDAR_NUM,
    'n_hidden_layers':4,
    'n_action':len(discrete_actions),
    'discrete_actions':discrete_actions
}