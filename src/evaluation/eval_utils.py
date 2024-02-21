import sys
sys.path.append("..")
sys.path.append(".")
import time
VISUALIZE = False
if not VISUALIZE:
    import os
    # os.environ["SDL_VIDEODRIVER"]="dummy"
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from typing import DefaultDict
import pickle
SAVE_LOG = False

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


def draw_map(map, ):
    import matplotlib.pyplot as plt
    import os
    # fig=plt.figure()
    # ax=fig.add_subplot(111)
    fig, ax = plt.subplots()
    # init_button()

    ax.set_xlim(map.xmin,map.xmax)
    ax.set_ylim(map.ymin,map.ymax)
    plt.axis('off')
    ax.add_patch(plt.Polygon(xy=list(map.dest.create_box().coords)[:-1], color='b'))
    for obs in map.obstacles:
        ax.add_patch(plt.Polygon(xy=list(obs.shape.coords), color='gray'))
    ax.add_patch(plt.Polygon(xy=list(map.start.create_box().coords)[:-1], color='g'))

    path = './log/figure/dlp/'
    num_files = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
    fig = plt.gcf()
    fig.savefig(path+f'image_{num_files}.png')

    # plt.show()
    ax.clear()

def eval(env, agent:ParkingAgent, episode=2000, log_path='', multi_level=False, post_proc_action=True):
    succ_rate_case = DefaultDict(list)
    if multi_level:
        succ_rate_level = DefaultDict(list)
        step_num_level = DefaultDict(list)
        path_length_level = DefaultDict(list)
    reward_case = DefaultDict(list)
    reward_record = []
    succ_record = []
    success_step_record = []
    step_record = DefaultDict(list)
    path_length_record = DefaultDict(list)
    eval_record = []
    for i in trange(episode):
        obs = env.reset(i+1)
        agent.reset()
        # draw_map(env.map)
        # continue
        # case_id_list.append()
        done = False
        total_reward = 0
        step_num = 0
        path_length = 0
        last_xy = (env.vehicle.state.loc.x, env.vehicle.state.loc.y)
        last_obs = obs['target']
        while not done:
            # t = time.time()
            step_num += 1
            # t = time.time()
            # action = env.action_space.sample()
            if post_proc_action:
                action, _ = agent.choose_action(obs) #  agent.get_action(obs)
            else:
                action, _ = agent.get_action(obs)
            # print(time.time()-t)
            if (last_obs == obs['target']).all():
                action = env.action_space.sample()
            last_obs = obs['target']
            # action = [0,0]
            next_obs, reward, done, info = env.step(action)
            total_reward += reward
            obs = next_obs
            path_length += np.linalg.norm(np.array(last_xy)-np.array((env.vehicle.state.loc.x, env.vehicle.state.loc.y)))
            last_xy = (env.vehicle.state.loc.x, env.vehicle.state.loc.y)
            
            if info['path_to_dest'] is not None:
                agent.set_planner_path(info['path_to_dest'])
                # done = True
                # info['status']=Status.ARRIVED
                # step_record[env.map.case_id].append(step_num)
                # succ_record.append(1)
                # break
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
        if step_num < 100:
            path_length_record[env.map.case_id].append(path_length)
        reward_case[env.map.case_id].append(reward_record[-1])
        if multi_level:
            succ_rate_level[env.map.map_level].append(succ_record[-1])
            if step_num < 100:
                path_length_level[env.map.map_level].append(path_length)
            step_num_level[env.map.map_level].append(step_num)
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
                            'path_length':path_length,
                            })
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

    if multi_level:
        print('success rate per level: ')
        for k in succ_rate_level.keys():
            print('%s (case num %s):'%(k, len(succ_rate_level[k])) + '%s '%np.mean(succ_rate_level[k]))
    
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
        f_record_txt.write('step num: %s '%np.mean(success_step_record)+'+-(%s)\n'%np.std(success_step_record))
        if multi_level:
            f_record_txt.write('\n')
            for k in succ_rate_level.keys():
                f_record_txt.write('%s (case num %s):'%(k, len(succ_rate_level[k])) + '%s \n'%np.mean(succ_rate_level[k]))
                f_record_txt.write('step num: %s '%np.mean(step_num_level[k])+'+-(%s)\n'%np.std(step_num_level[k]))
                f_record_txt.write('path length: %s '%np.mean(path_length_level[k])+'+-(%s)\n'%np.std(path_length_level[k]))
        if len(case_ids) < 10:
            for k in case_ids:
                f_record_txt.write('\ncase %s : '%k + 'success rate: %s \n'%np.mean(succ_rate_case[k]))
                f_record_txt.write('step num: %s '%np.mean(step_record[k])+'+-(%s)\n'%np.std(step_record[k]))
                f_record_txt.write('path length: %s '%np.mean(path_length_record[k])+'+-(%s)\n'%np.std(path_length_record[k]))
        f_record_txt.close()
    
    return np.mean(succ_record)




if __name__=="__main__":
    verbose = False

    if VISUALIZE:
        raw_env = CarParking(fps=20, verbose=verbose, )#render_mode='rgb_array')
    else:
        raw_env = CarParking(fps=100, verbose=verbose,render_mode='rgb_array')
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

    agent = PPO(configs)
    checkpoint_path = None#'./log/eval_model/PPO2_mixed_99999.pt'
    if checkpoint_path is not None:
        agent.load(checkpoint_path,params_only=True)
        # agent = torch.load(checkpoint_path)
        print('load pre-trained model!')
    img_encoder_dir = ["./log/ae/20230822_181148/ae_True.pt", './log/ae_True.pt'][0]
    img_encoder_checkpoint = img_encoder_dir if USE_IMG else None
    if img_encoder_checkpoint is not None:
        agent.load_img_encoder(img_encoder_checkpoint, require_grad=UPDATE_IMG_ENCODE)
    # agent.action_filter.load('./log/af_100000.pt')

    step_ratio = env.vehicle.kinetic_model.step_len*env.vehicle.kinetic_model.n_step*VALID_SPEED[1]
    rs_planner = RsPlanner(step_ratio)
    parking_agent = ParkingAgent(agent, rs_planner)

    t = time.time()
    current_time = time.localtime()
    timestamp = time.strftime("%Y%m%d_%H%M%S", current_time)
    save_path = './log/eval_result/%s' % timestamp if SAVE_LOG else None
    if SAVE_LOG and not os.path.exists(save_path):
        os.makedirs(save_path)
    success_rate = eval(env, parking_agent, log_path=save_path)
    print('eval time: ',time.time()-t)

    env.close()
    # print(stop_reason_cnt)