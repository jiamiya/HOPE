
import os
# os.environ["SDL_VIDEODRIVER"]="dummy"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import numpy as np
import torch
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
SEED = 42

#########################
# vehicle
WHEEL_BASE = 2.8  # wheelbase
FRONT_HANG = 0.96  # front hang length
REAR_HANG = 0.93  # rear hang length
LENGTH = WHEEL_BASE+FRONT_HANG+REAR_HANG
WIDTH = 1.94  # width


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
VALID_STEER = [-0.75, 0.75]
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
# if MAP_LEVEL in ['Normal', 'Complex', 'Extrem',]:
#     MIN_PARA_PARK_LOT_LEN = MIN_PARK_LOT_LEN_DICT[MAP_LEVEL]
#     MAX_PARA_PARK_LOT_LEN =  MAX_PARK_LOT_LEN_DICT[MAP_LEVEL]
    # MIN_BAY_PARK_LOT_WIDTH = WIDTH + 0.85 # 1.2
    # the distance that the obstacles out of driving area is from dest 
    # BAY_PARK_WALL_DIST = 7.0 
    # PARA_PARK_WALL_DIST = 4.5
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
# START_COLOR = (10, 10, 200, 255)
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
P_RS = 1 # the probability to search RS path each training step
RS_MAX_DIST = 10
DRAW_DEST_DIRECTION = False
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

vehicle_base = np.array(
    [3.76, 3.765160020952183, 3.7807111311588217, 3.806868872962891, 3.8440006366925097, 3.7477922060015847, 3.138985938174796, 2.706715266336576, 2.3848355355070114, 2.1366085866477085, 1.9400000000000004, 1.7809961050133638, 1.6502625682029572, 1.5413452571937767, 1.4496422533686704, 1.3717871555019023, 1.3052637477181848, 1.2481567789163723, 1.198985938174796, 1.156592394050869, 1.120059522227874, 1.08865645050533, 1.0617971901508656, 1.0390106438279183, 1.0199183575111193, 1.0042178949977807, 0.9916703770190783, 0.9820911720143629, 0.9753430311766109, 0.9713311756179833, 0.97, 0.9713311756179833, 0.9753430311766109, 0.9820911720143628, 0.9916703770190783, 1.0042178949977805,
1.0199183575111193, 1.0390106438279183, 1.0617971901508652, 1.0886564505053298, 1.1200595222278737, 1.1565923940508687, 1.1989859381747958, 1.2481567789163721, 1.3052637477181843, 1.3152186130069785, 1.2514384385339299, 1.1966863962806455, 1.1495432190748047, 1.1088978623374313, 1.073871500692704, 1.0437634009999557, 1.0180117390106234, 0.996164844082437, 0.9778598685415886, 0.9628068477813773, 0.9507767532244773, 0.9415925669828428, 0.9351226999940703, 0.9312762817780665, 0.93, 0.9312762817780665, 0.9351226999940703, 0.9415925669828428, 0.9507767532244773, 0.9628068477813772, 0.9778598685415886, 0.9961648440824369, 1.0180117390106236, 1.0437634009999555, 1.0738715006927038,
1.108897862337431, 1.1495432190748043, 1.1966863962806453, 1.2514384385339299, 1.315218613006978, 1.3052637477181852, 1.2481567789163728, 1.1989859381747963, 1.1565923940508689, 1.1200595222278744, 1.08865645050533, 1.0617971901508658, 1.039010643827918, 1.0199183575111193, 1.0042178949977805, 0.9916703770190785, 0.9820911720143629, 0.9753430311766109, 0.9713311756179833, 0.97, 0.9713311756179833, 0.9753430311766108, 0.9820911720143628, 0.9916703770190785, 1.0042178949977807, 1.0199183575111193, 1.0390106438279179, 1.0617971901508652, 1.0886564505053298, 1.120059522227874, 1.1565923940508684, 1.1989859381747958, 1.2481567789163714, 1.305263747718185, 1.3717871555019017, 1.4496422533686706, 1.541345257193776, 1.6502625682029568, 1.780996105013361, 1.9400000000000015, 2.136608586647708, 2.3848355355070114, 2.706715266336572, 3.1389859381747938,
3.747792206001585, 3.8440006366925097, 3.806868872962891, 3.7807111311588217, 3.765160020952183]
)

CONFIGS_ACTION = {
    'use_tanh_activate': True,
    'hidden_size':256,
    'lidar_shape':LIDAR_NUM,
    'n_hidden_layers':4,
    'n_action':len(discrete_actions),
    'discrete_actions':discrete_actions
}