import sys
sys.path.append("..")
sys.path.append(".")
import time
VISUALIZE = False
if not VISUALIZE:
    import os
    os.environ["SDL_VIDEODRIVER"]="dummy"
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from typing import DefaultDict
import pickle
SAVE_LOG = True

import numpy as np
import torch
from tqdm import trange

# from model.MultiModalPPO_AF import PPO
from model.agent.ppo_agent import PPOAgent as PPO
from model.agent.parking_agent import ParkingAgent, RsPlanner
from env.car_parking_base import CarParking
from env.env_wrapper import CarParkingWrapper
from env.vehicle import VALID_SPEED,Status
from env.map_level import get_map_level
from configs import *
from dataset import ParkingDatasetGenerator

RECORD_IMG = True
use_img_path = "use_img_no_tail" if RECORD_IMG else "no_img"
parking_dataset = ParkingDatasetGenerator(save_path="./parking/%s"%use_img_path)


def execute_rs_path(rs_path, agent:PPO, env, obs):
    action_type = {'L':1, 'S':0, 'R':-1}
    step_ratio = env.vehicle.kinetic_model.step_len*env.vehicle.kinetic_model.n_step*VALID_SPEED[1]
    action_list = []
    for i in range(len(rs_path.ctypes)):
        steer = action_type[rs_path.ctypes[i]]
        step_len = rs_path.lengths[i]/step_ratio
        action_list.append([steer, step_len])

    # divide the action
    filtered_actions = []
    for action in action_list:
        action[0] *= 1
        if abs(action[1])<1 and abs(action[1])>1e-3:
            filtered_actions.append(action)
        elif action[1]>1:
            while action[1]>1:
                filtered_actions.append([action[0], 1])
                action[1] -= 1
            if abs(action[1])>1e-3:
                filtered_actions.append(action)
        elif action[1]<-1:
            while action[1]<-1:
                filtered_actions.append([action[0], -1])
                action[1] += 1
            if abs(action[1])>1e-3:
                filtered_actions.append(action)

    # step actions
    total_reward = 0
    for action in filtered_actions:
        next_obs, reward, done, info = env.step(action)
        parking_dataset.record(obs, action, reward, next_obs, done, info, rs=True)
        total_reward += reward
        obs = next_obs
        if done:
            break

    return total_reward

def eval(env, agent, episode=2000, log_path=''):
    succ_rate_case = DefaultDict(list)
    reward_case = DefaultDict(list)
    reward_record = []
    succ_record = []
    success_step_record = []
    step_record = DefaultDict(list)
    eval_record = []
    for i in trange(episode):
        obs = env.reset(i+1)
        parking_dataset.reset(env.map, env.level)
        # draw_map(env.map)
        # continue
        # case_id_list.append()
        done = False
        total_reward = 0
        step_num = 0
        last_obs = obs['target']
        while not done:
            # t = time.time()
            step_num += 1
            # action, _ = agent.get_action(obs)
            action, _ = agent.choose_action(obs) #  agent.get_action(obs)
            if (last_obs == obs['target']).all():
                action = env.action_space.sample()
            last_obs = obs['target']
            # action = [0,0]
            next_obs, reward, done, info = env.step(action)
            parking_dataset.record(obs, action, reward, next_obs, done, info, rs=False)
            total_reward += reward
            obs = next_obs
            
            if info['path_to_dest'] is not None:
                succ_record.append(1)
                rs_reward = execute_rs_path(info['path_to_dest'], agent, env, obs)
                total_reward += rs_reward
                # step_record[env.map.case_id].append(step_num)
                break
            # else:
            #     succ_record.append(0)
            #     break
            if done:
                # print(info['status'])
                if info['status']==Status.ARRIVED:
                    succ_record.append(1)
                else:
                    succ_record.append(0)
        reward_record.append(total_reward)
        succ_rate_case[env.map.case_id].append(succ_record[-1])
        reward_case[env.map.case_id].append(reward_record[-1])
        if info['status']==Status.OUTBOUND:
            step_record[env.map.case_id].append(200)
        else:
            step_record[env.map.case_id].append(step_num)
        if succ_record[-1] == 1:
            success_step_record.append(step_num)
        eval_record.append({'case_id':env.map.case_id,
                            'status':info['status'],
                            'step_num':step_num,
                            'reward':total_reward,
                            })
        
        # other info to save
        other_info = {
            'success': succ_record[-1],
            'status': info['status'],
            'total_reward': total_reward,
                            }
        parking_dataset.save_to_disk(other_info=other_info)

    print('#'*15)
    print('EVALUATE RESULT:')
    print('success rate: ', np.mean(succ_record))
    print('average reward: ', np.mean(reward_record))
    print('-'*10)
    print('success rate per case: ')
    case_ids = [int(k) for k in succ_rate_case.keys()]
    case_ids.sort()
    if len(case_ids) < 10:
        print('-'*10)
        print('average reward per case: ')
        for k in case_ids:
            env.reset(k)
            print('case %s (%s) :'%(k,get_map_level(env.map.start, env.map.dest, env.map.obstacles))\
                , np.mean(succ_rate_case[k]))
        for k in case_ids:
            print('case %s :'%k, np.mean(reward_case[k]), np.mean(step_record[k]), '+-(%s)'%np.std(step_record[k]))
    
    if log_path is not None:
        def plot_time_ratio(node_list):
            max_node = TOLERANT_TIME
            raw_len = len(node_list)
            filtered_node_list = []
            for n in node_list:
                if n != max_node:
                    filtered_node_list.append(n)
            filtered_node_list.sort()
            ratio_list = [i/raw_len for i in range(1,len(filtered_node_list)+1)]
            # ratio_list[-1] = ratio_list[-2]
            import matplotlib.pyplot as plt
            plt.plot(filtered_node_list, ratio_list)
            plt.xlabel('Search node')
            plt.ylabel('Accumulate success rate')
            # plt.xlim(right=20000-10)
            fig = plt.gcf()
            fig.savefig(log_path+'/success_rate.png')
            plt.close()
        all_step_record = []
        for k in step_record.keys():
            all_step_record.extend(step_record[k])
        plot_time_ratio(all_step_record)

        # save eval result
        f_record = open(log_path+'/record.data', 'wb')
        pickle.dump(eval_record, f_record)
        f_record.close()

        # f_record = open(log_path+'/record.data', 'rb')
        # data = pickle.load(f_record)
        # print(data)

        f_record_txt = open(log_path+'/result.txt', 'w', newline='')
        f_record_txt.write('success rate: %s\n'%np.mean(succ_record))
        f_record_txt.write('step num: %s '%np.mean(success_step_record)+'+-(%s)'%np.std(success_step_record))
        f_record_txt.close()
    
    return np.mean(succ_record)




if __name__=="__main__":
    verbose = False

    if VISUALIZE:
        raw_env = CarParking(fps=20, verbose=verbose, use_img_observation=RECORD_IMG)#render_mode='rgb_array')
    else:
        raw_env = CarParking(fps=100, verbose=verbose,render_mode='rgb_array', use_img_observation=RECORD_IMG)
    env = CarParkingWrapper(raw_env)

    act_dim = env.action_space.shape[0]
    stop_reason_cnt = DefaultDict(int)

    seed = 42
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

    rl_agent = PPO(configs)
    checkpoint_path = './data_collection/pretrain_agent_no_img.pt'
    if checkpoint_path is not None:
        rl_agent.load(checkpoint_path,params_only=True)
        # agent = torch.load(checkpoint_path)
        print('load pre-trained model!')
    img_encoder_checkpoint = './log/ae_True.pt' if USE_IMG else None
    if img_encoder_checkpoint is not None:
        rl_agent.load_img_encoder(img_encoder_checkpoint, require_grad=UPDATE_IMG_ENCODE)
    # agent.action_filter.load('./log/af_100000.pt')
    step_ratio = env.vehicle.kinetic_model.step_len*env.vehicle.kinetic_model.n_step*VALID_SPEED[1]
    rs_planner = RsPlanner(step_ratio)
    agent = ParkingAgent(rl_agent, rs_planner)

    t = time.time()
    current_time = time.localtime()
    timestamp = time.strftime("%Y%m%d_%H%M%S", current_time)
    save_path = './log/eval_result/%s' % timestamp if SAVE_LOG else None
    # if SAVE_LOG and not os.path.exists(save_path):
    #     os.makedirs(save_path)
    # success_rate = eval(env, agent, log_path=save_path)
    # print('eval time: ',time.time()-t)

    env.set_level('dlp')
    log_path = save_path+'/dlp'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    eval(env, agent, episode=3000, log_path=log_path)
    
    # eval on extreme
    env.set_level('Extrem')
    log_path = save_path+'/extreme'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    eval(env, agent, episode=500, log_path=log_path)
    
    # eval on complex
    env.set_level('Complex')
    log_path = save_path+'/complex'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    eval(env, agent, episode=500, log_path=log_path)
    
    # eval on normalize
    env.set_level('Normal')
    log_path = save_path+'/normalize'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    eval(env, agent, episode=2000, log_path=log_path)

    env.close()
    # print(stop_reason_cnt)