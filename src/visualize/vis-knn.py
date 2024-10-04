import sys
sys.path.append("..")
sys.path.append(".")
import time
import os
import argparse
import pickle as pkl
from copy import deepcopy

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

# from model.MultiModalPPO_AF import PPO
from model.agent.ppo_agent import PPOAgent as PPO
from model.agent.sac_agent import SACAgent as SAC
from model.agent.parking_agent import ParkingAgent, RsPlanner
from env.car_parking_base import CarParking
from env.env_wrapper import CarParkingWrapper
from env.vehicle import VALID_SPEED, Status, State
from evaluation.eval_utils import eval
from configs import *
from visualize.vis_utils import draw_map

data_path = './log/eval/20240929_183243/data_analysis.pkl'
# data_path = r'.\log\eval\20240929_190442\data_analysis.pkl'

label_map = {
    'Normal0': 'vertical',
    'Normal1': 'parallel',
    'Complex0': 'vertical',
    'Complex1': 'parallel',
    'Extrem1': 'parallel',
    'dlp': 'dlp'
}


def run_and_collect_data(env, agent, episode=100,  post_proc_action=True, data=[]):
    succ_record = []
    for i in range(episode):
        obs = env.reset(i+1)
        agent.reset()
        done = False
        total_reward = 0
        step_num = 0
        path_length = 0
        last_xy = (env.vehicle.state.loc.x, env.vehicle.state.loc.y)
        last_obs = obs['target']
        while not done:
            step_num += 1
            if post_proc_action:
                action, _ = agent.choose_action(obs)
            else:
                action, _ = agent.get_action(obs)
            if (last_obs == obs['target']).all():
                action = env.action_space.sample()
            scen_type = str(env.map_type) + str(env.map.case_id) if env.map_type != 'dlp' else 'dlp'
            scen_type = scen_type + '_test'
            if step_num%5 == 0:
                data_to_save = {
                    'obs': obs,
                    'action': action,
                    'embedding': agent.actor_net.current_embedding,
                    'scen_type': scen_type,
                    'map': deepcopy(env.map),
                    'trajectory': deepcopy(env.vehicle.trajectory)
                }
                data.append(data_to_save)
            last_obs = obs['target']
            next_obs, reward, done, info = env.step(action)
            total_reward += reward
            obs = next_obs
            path_length += np.linalg.norm(np.array(last_xy)-np.array((env.vehicle.state.loc.x, env.vehicle.state.loc.y)))
            last_xy = (env.vehicle.state.loc.x, env.vehicle.state.loc.y)
            
            if info['path_to_dest'] is not None:
                agent.set_planner_path(info['path_to_dest'])
            if done:
                if info['status']==Status.ARRIVED:
                    succ_record.append(1)
                else:
                    succ_record.append(0)
    return data

def _pseudo_env_step(env, vehicle_pose):
    
        prev_state = env.vehicle.state
        collide = False
        arrive = False

        env.vehicle.state = State(vehicle_pose)
        env.vehicle.box = env.vehicle.state.create_box()
        env.vehicle.trajectory.append(env.vehicle.state)
        env.vehicle.tmp_trajectory.append(env.vehicle.state)

        env.t += 1
        observation = env.render(env.render_mode)
        if arrive:
            status = Status.ARRIVED
        else:
            status = Status.CONTINUE

        reward_list = env.get_reward(status, prev_state)
        reward_info = OrderedDict({'time_cost':reward_list[0],\
            'rs_dist_reward':reward_list[1],\
            'dist_reward':reward_list[2],\
            'angle_reward':reward_list[3],\
            'box_union_reward':reward_list[4],})

        info = OrderedDict({'reward_info':reward_info,
            'path_to_dest':None})
        
        returns = observation, reward_info, status, info
        obs, reward, status, info = env.reward_func(*returns)
        obs = env.obs_func(obs)
        done = False if status==Status.CONTINUE else True

        return obs, reward, done, info

def run_recorded_data(env, agent, episode=100,  post_proc_action=True, data=[]):
    succ_record = []
    for i in range(episode):
        obs = env.reset(i+1)
        agent.reset()
        done = False
        total_reward = 0
        step_num = 0
        path_length = 0
        last_xy = (env.vehicle.state.loc.x, env.vehicle.state.loc.y)
        last_obs = obs['target']
        ref_traj = env.map.data['ref_traj']
        print('ref_traj:', ref_traj)
        print('dest:', env.map.data['goal'])
        while not done:
            step_num += 1
            if post_proc_action:
                action, _ = agent.choose_action(obs)
            else:
                action, _ = agent.get_action(obs)
            if (last_obs == obs['target']).all():
                action = env.action_space.sample()
            scen_type = str(env.map_type) + str(env.map.case_id) if env.map_type != 'dlp' else 'dlp'
            scen_type = scen_type + '_test'
            if step_num%2 == 0:
                data_to_save = {
                    'obs': obs,
                    'action': action,
                    'embedding': agent.actor_net.current_embedding,
                    'scen_type': scen_type,
                    'map': deepcopy(env.map),
                    'trajectory': deepcopy(env.vehicle.trajectory)
                }
                data.append(data_to_save)
            last_obs = obs['target']
            # next_obs, reward, done, info = env.step(action)
            next_pose = ref_traj[step_num-1]
            next_obs, reward, done, info = _pseudo_env_step(env, next_pose)
            if step_num == len(ref_traj):
                done = True
            total_reward += reward
            obs = next_obs
            path_length += np.linalg.norm(np.array(last_xy)-np.array((env.vehicle.state.loc.x, env.vehicle.state.loc.y)))
            last_xy = (env.vehicle.state.loc.x, env.vehicle.state.loc.y)
            
            if info['path_to_dest'] is not None:
                agent.set_planner_path(info['path_to_dest'])
            if done:
                if info['status']==Status.ARRIVED:
                    succ_record.append(1)
                else:
                    succ_record.append(0)
    return data




if __name__=="__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default='./model/ckpt/HOPE_SAC0.pt') # './model/ckpt/HOPE_SAC0.pt'
    parser.add_argument('--eval_episode', type=int, default=100)
    parser.add_argument('--verbose', type=bool, default=True)
    parser.add_argument('--visualize', type=bool, default=True)
    args = parser.parse_args()

    checkpoint_path = args.ckpt_path
    print('ckpt path: ',checkpoint_path)
    verbose = args.verbose

    if args.visualize:
        raw_env = CarParking(fps=100, verbose=verbose)
    else:
        raw_env = CarParking(fps=100, verbose=verbose, render_mode='rgb_array')
    env = CarParkingWrapper(raw_env)

    relative_path = '.'
    current_time = time.localtime()
    timestamp = time.strftime("%Y%m%d_%H%M%S", current_time)
    save_path = relative_path+'/log/eval/%s/' % timestamp
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    configs_file = os.path.join(save_path, 'configs.txt')
    with open(configs_file, 'w') as f:
        f.write(str(checkpoint_path))
    Agent_type = PPO if 'ppo' in checkpoint_path.lower() else SAC
    # writer = SummaryWriter(save_path)
    print("You can track the training process by command 'tensorboard --log-dir %s'" % save_path)

    seed = SEED
    # env.seed(seed)
    env.action_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

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
    if checkpoint_path is not None:
        rl_agent.load(checkpoint_path, params_only=True)
        print('load pre-trained model!')

    step_ratio = env.vehicle.kinetic_model.step_len*env.vehicle.kinetic_model.n_step*VALID_SPEED[1]
    rs_planner = RsPlanner(step_ratio)
    parking_agent = ParkingAgent(rl_agent, rs_planner)

    eval_episode = 1
    choose_action = True if isinstance(rl_agent, PPO) else False
    data_analysis = []
    with torch.no_grad():
        # eval on extreme
        # env.set_level('Extrem')
        # data_analysis = run_and_collect_data(env, parking_agent, episode=eval_episode, post_proc_action=choose_action, data=data_analysis)

        # # eval on dlp
        # env.set_level('dlp')
        # data_analysis = run_and_collect_data(env, parking_agent, episode=eval_episode, post_proc_action=choose_action, data=data_analysis)
        
        # # eval on complex
        # env.set_level('Complex')
        # data_analysis = run_and_collect_data(env, parking_agent, episode=eval_episode, post_proc_action=choose_action, data=data_analysis)
        
        # eval on normalize
        # env.set_level('Normal')
        # data_analysis = run_and_collect_data(env, parking_agent, episode=eval_episode, post_proc_action=choose_action, data=data_analysis)

        # eval on grid
        env.set_level('grid')
        data_analysis = run_recorded_data(env, parking_agent, episode=eval_episode, post_proc_action=choose_action, data=data_analysis)



    from sklearn.manifold import TSNE
    from sklearn.preprocessing import LabelEncoder
    from sklearn.neighbors import NearestNeighbors
    import matplotlib.pyplot as plt
    print('data_analysis:', len(data_analysis))
    t = time.time()
    data_collected = pkl.load(open(data_path, 'rb'))
    data_all = data_collected + data_analysis
    print('data_all:', len(data_all), 'time:', time.time()-t)

    # print(data_analysis[i][2])
    # print(data_analysis[i][2][0].shape)
    # for i in range(len(data_analysis)):
    #     print(data_analysis[i])
    #     print(data_analysis[i][2][0].shape)
    #     print(data_analysis[i][2][1].shape)
    # embeddings = [data_analysis[i]['embedding'][1].flatten() for i in range(len(data_analysis))]
    # embeddings = [data_all[i]['obs']['action_mask'] for i in range(len(data_all))]
    # embeddings = [data_all[i]['obs']['img'].flatten() for i in range(len(data_all))]
    # embeddings = [data_all[i]['embedding'][0][0][-1] for i in range(len(data_all))]  # the img embedding
    # embeddings = [data_all[i]['embedding'][0][0][-2] for i in range(len(data_all))]  # the action mask embedding
    embeddings = [data_all[i]['embedding'][1].flatten() for i in range(len(data_all))]  # the fused embedding

    def _prepocess_label(label:str):
        if label in label_map:
            return label_map[label]
        else:
            return label
    labels = [_prepocess_label(data_all[i]['scen_type']) for i in range(len(data_all))]
    print(embeddings[0].shape, len(embeddings))
    embeddings = np.array(embeddings)

    # imgs = [data_all[i][0]['img'] for i in range(len(data_all))]
    # img_0 = imgs[0]
    # print(img_0.shape)
    # plt.imshow(img_0.transpose(1,2,0))
    # plt.show()
    # exit()

    print('start knn')
    t = time.time()
    knn = NearestNeighbors(n_neighbors=3)
    knn.fit(embeddings[:len(data_collected)])
    print('knn fit done, time:', time.time()-t)

    for i in range(len(data_analysis)):
        dist, idx = knn.kneighbors(embeddings[len(data_collected)+i].reshape(1,-1))
        print(embeddings[len(data_collected)+i])
        print(embeddings[idx[0][0]])
        print(embeddings[idx[0][1]])
        print('dist:', dist)
        print('idx:', idx)
        similar_imgs = [data_all[j]['obs']['img'].transpose(1,2,0) for j in idx[0]]
        map_imgs = [draw_map(data_all[j]['map'], traj=data_all[j]['trajectory'], save_path=None, draw_history=False) for j in idx[0]]

        for idx_similar in idx[0]:
            labels[idx_similar] = labels[idx_similar]+'_sim' if not labels[idx_similar].endswith('sim') else labels[idx_similar]
        # show the raw img and the similar imgs in one fig
        plt.figure()
        map_img = draw_map(data_all[len(data_collected)+i]['map'],\
             traj=data_all[len(data_collected)+i]['trajectory'], save_path=None, draw_history=False)
        plt.subplot(2,2,1)
        plt.axis('off')
        # plt.imshow(data_all[len(data_collected)+i]['obs']['img'].transpose(1,2,0))
        plt.imshow(map_img)
        plt.title('current scen')
        for j in range(3):
            plt.subplot(2,2,j+2)
            plt.axis('off')
            plt.imshow(map_imgs[j])
            plt.title('similar scen %d' % j)
        plt.savefig(save_path+'similar_scen_%d.png' % i, dpi=600)
        plt.show()



    le = LabelEncoder()
    numeric_labels = le.fit_transform(labels)

    # 生成颜色映射
    num_classes = len(np.unique(numeric_labels))
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes)) 

    tsne = TSNE(n_components=2, random_state=0, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings)

    print(embeddings_2d.shape)

    for i, label in enumerate(le.classes_):
        marker = 'x' if label.endswith('test') else '.'
        color = colors[i] if not label.endswith('test') else 'red'
        plt.scatter(embeddings_2d[numeric_labels == i, 0], 
                embeddings_2d[numeric_labels == i, 1],
                marker=marker,
                color=color, 
                label=label)
    for i in range(len(data_analysis)-1):
        # plot the transition arrow of the sequence of data_analysis
        idx_analy = len(data_collected)+i
        plt.arrow(embeddings_2d[idx_analy,0], embeddings_2d[idx_analy,1],\
         embeddings_2d[idx_analy+1,0]-embeddings_2d[idx_analy,0], embeddings_2d[idx_analy+1,1]-embeddings_2d[idx_analy,1],\
              width=0.01, color='black', zorder=10, head_width=0.1, head_length=0.1)
        
    plt.legend()
    plt.savefig(save_path+'tsne.png', dpi=600)
    plt.show()



    env.close()