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
from tqdm import trange

# from model.MultiModalPPO_AF import PPO
from model.agent.ppo_agent import PPOAgent as PPO
from model.agent.sac_agent import SACAgent as SAC
from model.agent.parking_agent import ParkingAgent, RsPlanner
from env.car_parking_base import CarParking
from env.env_wrapper import CarParkingWrapper
from env.vehicle import VALID_SPEED, Status
from evaluation.eval_utils import eval
from configs import *

def run_and_collect_data(env, agent, episode=100,  post_proc_action=True, data=[]):
    succ_record = []
    for i in trange(episode):
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


if __name__=="__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default='./model/ckpt/HOPE_SAC0.pt') # './model/ckpt/HOPE_SAC0.pt'
    parser.add_argument('--eval_episode', type=int, default=10)
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
    writer = SummaryWriter(save_path)
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
        env.set_level('Extrem')
        data_analysis = run_and_collect_data(env, parking_agent, episode=eval_episode, post_proc_action=choose_action, data=data_analysis)

        # eval on dlp
        env.set_level('dlp')
        data_analysis = run_and_collect_data(env, parking_agent, episode=eval_episode, post_proc_action=choose_action, data=data_analysis)
        
        # eval on complex
        env.set_level('Complex')
        data_analysis = run_and_collect_data(env, parking_agent, episode=eval_episode, post_proc_action=choose_action, data=data_analysis)
        
        # eval on normalize
        env.set_level('Normal')
        data_analysis = run_and_collect_data(env, parking_agent, episode=eval_episode, post_proc_action=choose_action, data=data_analysis)

    data_analysis_path = save_path + 'data_analysis.pkl'
    with open(data_analysis_path, 'wb') as f:
        pkl.dump(data_analysis, f)

    print(len(data_analysis))

    from sklearn.manifold import TSNE
    from sklearn.preprocessing import LabelEncoder
    import matplotlib.pyplot as plt

    # print(data_analysis[i][2])
    # print(data_analysis[i][2][0].shape)
    # for i in range(len(data_analysis)):
    #     print(data_analysis[i])
    #     print(data_analysis[i][2][0].shape)
    #     print(data_analysis[i][2][1].shape)
    # embeddings = [data_analysis[i][2][0].flatten() for i in range(len(data_analysis))]
    embeddings = [data_analysis[i]['obs']['action_mask'] for i in range(len(data_analysis))]
    labels = [data_analysis[i]['scen_type'] for i in range(len(data_analysis))]
    print(embeddings[0].shape, len(embeddings))
    embeddings = np.array(embeddings)

    from visualize.vis_utils import draw_map
    example = data_analysis[1]
    map = example['map']
    draw_map(map, traj=example['trajectory'], save_path=save_path+'map.png')


    le = LabelEncoder()
    numeric_labels = le.fit_transform(labels)

    # 生成颜色映射
    num_classes = len(np.unique(numeric_labels))
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes)) 

    tsne = TSNE(n_components=2, random_state=0, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings)

    print(embeddings_2d.shape)

    for i, label in enumerate(le.classes_):
        plt.scatter(embeddings_2d[numeric_labels == i, 0], 
                embeddings_2d[numeric_labels == i, 1], 
                color=colors[i], 
                label=label)
    plt.legend()
    plt.savefig(save_path+'tsne.png')
    plt.show()



    env.close()