import sys
sys.path.append("..")
sys.path.append(".")
import time
import os
# os.environ["SDL_VIDEODRIVER"]="dummy"
# os.environ["CUDA_VISIBLE_DEVICES"] = '3'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from typing import DefaultDict
import argparse

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter

# from model.MultiModalPPO_AF import PPO
from model.agent.ppo_agent import PPOAgent as PPO
from model.agent.sac_agent import SACAgent as SAC
from model.agent.parking_agent import ParkingAgent, RsPlanner
from env.car_parking_base import CarParking
from env.env_wrapper import CarParkingWrapper
from env.vehicle import VALID_SPEED,Status
from evaluation.eval_utils import eval
from configs import *

IMG_ENCODER_PATH = ["./log/ae/20230825_132652/ae_True.pt", './log/ae/ae_True.pt'][1]
USE_RS_PLANNER = True


if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Demo of argparse')
 
    parser.add_argument('ckpt_path', type=str)
    args = parser.parse_args()
    checkpoint_path = './log' + args.ckpt_path.split('log')[-1]
    # checkpoint_path = args.ckpt_path.split('draft')[-1]
    print('ckpt path: ',checkpoint_path)

    save = False
    verbose = False

    raw_env = CarParking(fps=100, verbose=verbose,)#render_mode='rgb_array')
    env = CarParkingWrapper(raw_env)

    relative_path = '.'#os.path.dirname(os.getcwd())
    current_time = time.localtime()
    timestamp = time.strftime("%Y%m%d_%H%M%S", current_time)
    save_path = relative_path+'/log/eval/ppo_mixed/%s/' % timestamp
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    configs_file = os.path.join(save_path, 'configs.txt')
    with open(configs_file, 'w') as f:
        f.write(str(checkpoint_path))
    Agent_type = PPO if 'ppo' in checkpoint_path.lower() else SAC

    writer = SummaryWriter(save_path)
    print("You can track the training process by command 'tensorboard --log-dir %s'" % save_path)

    act_dim = env.action_space.shape[0]
    stop_reason_cnt = DefaultDict(int)

    seed = SEED
    env.seed(seed)
    env.action_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print('use_lidar_observation:', env.use_lidar_observation)
    feature_size1 = env.lidar.lidar_num
    feature_size2 = env.tgt_repr_size
    EMBD_SIZE = 128
    HIDD_SIZE = 256
    actor_params = ACTOR_CONFIGS
    critic_params = CRITIC_CONFIGS

    configs = {
        "discrete": False,
        "observation_shape": env.observation_shape,
        "action_dim": env.action_space.shape[0],
        "hidden_size": 64,
        "activation": "tanh",
        "dist_type": "gaussian",
        "save_params": False,
        "actor_layers": actor_params,
        "critic_layers": critic_params,
    }
    print('observation_space:',env.observation_space)

    rl_agent = Agent_type(configs)
    # checkpoint_path = './log/exp1/20231124_044713/PPO_best.pt' #/20230906_084855 draft\log\ppo_mixed\PPO2_49999.pt
    if checkpoint_path is not None:
        rl_agent.load(checkpoint_path, params_only=True)
        # agent = torch.load(checkpoint_path)
        print('load pre-trained model!')
    # img_encoder_checkpoint =  IMG_ENCODER_PATH if USE_IMG else None
    # if img_encoder_checkpoint is not None:
    #     rl_agent.load_img_encoder(img_encoder_checkpoint, require_grad=UPDATE_IMG_ENCODE)
    # agent.action_filter.load('./log/af_100000.pt')

    step_ratio = env.vehicle.kinetic_model.step_len*env.vehicle.kinetic_model.n_step*VALID_SPEED[1]
    rs_planner = RsPlanner(step_ratio) if USE_RS_PLANNER else None
    parking_agent = ParkingAgent(rl_agent, rs_planner)

    eval_episode = 1000
    choose_action = True if isinstance(rl_agent, PPO) else False
    # print(choose_action)
    with torch.no_grad():
        # eval on extreme
        env.set_level('Extrem')
        log_path = save_path+'/extreme'
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        eval(env, parking_agent, episode=eval_episode, log_path=log_path, post_proc_action=choose_action)

        # eval on dlp
        env.set_level('dlp')
        log_path = save_path+'/dlp'
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        eval(env, parking_agent, episode=eval_episode, log_path=log_path, multi_level=True, post_proc_action=choose_action)
        
        # eval on complex
        env.set_level('Complex')
        log_path = save_path+'/complex'
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        eval(env, parking_agent, episode=eval_episode, log_path=log_path, post_proc_action=choose_action)
        
        # eval on normalize
        env.set_level('Normal')
        log_path = save_path+'/normalize'
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        eval(env, parking_agent, episode=eval_episode, log_path=log_path, post_proc_action=choose_action)

        env.close()
        # print(stop_reason_cnt)