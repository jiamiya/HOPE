import sys
sys.path.append("..")
sys.path.append(".")
import time
import os
from shutil import copyfile
import argparse

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter

from model.agent.ppo_agent import PPOAgent as PPO
from model.agent.parking_agent import ParkingAgent, RsPlanner
from env.car_parking_base import CarParking
from env.env_wrapper import CarParkingWrapper
from env.vehicle import VALID_SPEED,Status
from evaluation.eval_utils import eval
from configs import *


class SceneChoose():
    def __init__(self) -> None:
        self.scene_types = {0:'Normal', 
                            1:'Complex',
                            2:'Extrem',
                            3:'dlp',
                            }
        self.target_success_rate = np.array([0.95, 0.95, 0.9, 0.99])
        self.success_record = {}
        for scene_name in self.scene_types:
            self.success_record[scene_name] = []
        self.scene_record = []
        self.history_horizon = 200
        
        
    def choose_case(self,):
        if len(self.scene_record) < self.history_horizon:
            scene_chosen = self._choose_case_uniform()
        else:
            if np.random.random() > 0.5:
                scene_chosen = self._choose_case_worst_perform()
            else:
                scene_chosen = self._choose_case_uniform()
        self.scene_record.append(scene_chosen)
        return self.scene_types[scene_chosen]
    
    def update_success_record(self, success:int):
        self.success_record[self.scene_record[-1]].append(success)

    def _choose_case_uniform(self,):
        case_count = np.zeros(len(self.scene_types))
        for i in range(min(len(self.scene_record), self.history_horizon)):
            scene_id = self.scene_record[-(i+1)]
            case_count[scene_id] += 1
        return np.argmin(case_count)
    
    def _choose_case_worst_perform(self,):
        success_rate = []
        for i in self.success_record.keys():
            idx = int(i)
            recent_success_record = self.success_record[idx][-min(250, len(self.success_record[idx])):]
            success_rate.append(np.sum(recent_success_record)/len(recent_success_record))
        fail_rate = self.target_success_rate - np.array(success_rate)
        fail_rate = np.clip(fail_rate, 0.01, 1)
        fail_rate = fail_rate/np.sum(fail_rate)
        return np.random.choice(np.arange(len(fail_rate)), p=fail_rate)

class DlpCaseChoose():
    def __init__(self) -> None:
        self.dlp_case_num = 248
        self.case_record = []
        self.case_success_rate = {}
        for i in range(self.dlp_case_num):
            self.case_success_rate[str(i)] = []
        self.horizon = 500
    
    def choose_case(self,):
        if np.random.random()<0.2 or len(self.case_record)<self.horizon:
            return np.random.randint(0, self.dlp_case_num)
        success_rate = []
        for i in range(self.dlp_case_num):
            idx = str(i)
            if len(self.case_success_rate[idx]) <= 1:
                success_rate.append(0)
            else:
                recent_success_record = self.case_success_rate[idx][-min(10, len(self.case_success_rate[idx])):]
                success_rate.append(np.sum(recent_success_record)/len(recent_success_record))
        fail_rate = 1-np.array(success_rate)
        fail_rate = np.clip(fail_rate, 0.005, 1)
        fail_rate = fail_rate/np.sum(fail_rate)
        return np.random.choice(np.arange(len(fail_rate)), p=fail_rate)
    
    def update_success_record(self, success:int, case_id:int):
        self.case_success_rate[str(case_id)].append(success)
        self.case_record.append(case_id)
        

if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent_ckpt', type=str, default=None) # './model/ckpt/PPO.pt'
    parser.add_argument('--img_ckpt', type=str, default='./model/ckpt/autoencoder.pt')
    parser.add_argument('--train_episode', type=int, default=100000)
    parser.add_argument('--eval_episode', type=int, default=2000)
    parser.add_argument('--verbose', type=bool, default=True)
    parser.add_argument('--visualize', type=bool, default=True)
    args = parser.parse_args()

    verbose = args.verbose

    if args.visualize:
        raw_env = CarParking(fps=100, verbose=verbose,)
    else:
        raw_env = CarParking(fps=100, verbose=verbose, render_mode='rgb_array')
    env = CarParkingWrapper(raw_env)
    scene_chooser = SceneChoose()
    dlp_case_chooser = DlpCaseChoose()

    # the path to log and save model
    relative_path = '.'
    current_time = time.localtime()
    timestamp = time.strftime("%Y%m%d_%H%M%S", current_time)
    save_path = relative_path+'/log/exp/ppo_%s/' % timestamp
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    writer = SummaryWriter(save_path)
    # configs log
    copyfile('./configs.py', save_path+'configs.txt')
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

    rl_agent = PPO(configs)
    checkpoint_path = args.agent_ckpt
    if checkpoint_path is not None:
        rl_agent.load(checkpoint_path, params_only=True)
        print('load pre-trained model!')
    img_encoder_checkpoint =  args.img_ckpt if USE_IMG else None
    if img_encoder_checkpoint is not None and os.path.exists(img_encoder_checkpoint):
        rl_agent.load_img_encoder(img_encoder_checkpoint, require_grad=UPDATE_IMG_ENCODE)
    else:
        print('not load img encoder')

    step_ratio = env.vehicle.kinetic_model.step_len*env.vehicle.kinetic_model.n_step*VALID_SPEED[1]
    rs_planner = RsPlanner(step_ratio)
    parking_agent = ParkingAgent(rl_agent, rs_planner)


    reward_list = []
    reward_per_state_list = []
    reward_info_list = []
    case_id_list = []
    succ_record = []
    best_success_rate = [0, 0, 0, 0]

    for i in range(args.train_episode):
        scene_chosen = scene_chooser.choose_case()
        if scene_chosen == 'dlp':
            case_id = dlp_case_chooser.choose_case()
        else:
            case_id = None
        obs = env.reset(case_id, None, scene_chosen)
        parking_agent.reset()
        case_id_list.append(env.map.case_id)
        done = False
        total_reward = 0
        step_num = 0
        reward_info = []
        xy = []
        while not done:
            step_num += 1
            action, log_prob = parking_agent.choose_action(obs)
            next_obs, reward, done, info = env.step(action)
            reward_info.append(list(info['reward_info'].values()))
            total_reward += reward
            reward_per_state_list.append(reward)
            parking_agent.push_memory((obs, action, reward, done, log_prob, next_obs))
            obs = next_obs
            if len(parking_agent.memory) % parking_agent.configs.batch_size == 0:
                if verbose:
                    print("Updating the agent.")
                actor_loss, critic_loss = parking_agent.update()
                writer.add_scalar("actor_loss", actor_loss, i)
                writer.add_scalar("critic_loss", critic_loss, i)
            
            if info['path_to_dest'] is not None:
                parking_agent.set_planner_path(info['path_to_dest'])

            if done:
                if info['status']==Status.ARRIVED:
                    succ_record.append(1)
                    scene_chooser.update_success_record(1)
                    if scene_chosen == 'dlp':
                        dlp_case_chooser.update_success_record(1, case_id)
                else:
                    succ_record.append(0)
                    scene_chooser.update_success_record(0)
                    if scene_chosen == 'dlp':
                        dlp_case_chooser.update_success_record(0, case_id)

            
        writer.add_scalar("total_reward", total_reward, i)
        writer.add_scalar("avg_reward", np.mean(reward_per_state_list[-1000:]), i)
        writer.add_scalar("action_std0", parking_agent.log_std.detach().cpu().numpy().reshape(-1)[0],i)
        writer.add_scalar("action_std1", parking_agent.log_std.detach().cpu().numpy().reshape(-1)[1],i)
        for type_id in scene_chooser.scene_types:
            writer.add_scalar("success_rate_%s"%scene_chooser.scene_types[type_id],
                np.mean(scene_chooser.success_record[type_id][-100:]), i)
        writer.add_scalar("step_num", step_num, i)
        reward_list.append(total_reward)
        reward_info = np.sum(np.array(reward_info), axis=0)
        reward_info = np.round(reward_info,2)
        reward_info_list.append(list(reward_info))

        if verbose and i%10==0 and i>0:
            print('success rate:',np.sum(succ_record),'/',len(succ_record))
            print(parking_agent.log_std.detach().cpu().numpy().reshape(-1))
            print("episode:%s  average reward:%s"%(i,np.mean(reward_list[-50:])))
            print(np.mean(parking_agent.actor_loss_list[-100:]),np.mean(parking_agent.critic_loss_list[-100:]))
            print('time_cost ,rs_dist_reward ,dist_reward ,angle_reward ,box_union_reward')
            for j in range(10):
                print(case_id_list[-(10-j)],reward_list[-(10-j)],reward_info_list[-(10-j)])
            print("")

        # save best model
        for type_id in scene_chooser.scene_types:
            success_rate_normal = np.mean(scene_chooser.success_record[0][-100:])
            success_rate_complex = np.mean(scene_chooser.success_record[1][-100:])
            success_rate_extreme = np.mean(scene_chooser.success_record[2][-100:])
            success_rate_dlp = np.mean(scene_chooser.success_record[3][-100:])
        if success_rate_normal >= best_success_rate[0] and success_rate_complex >= best_success_rate[1] and\
            success_rate_extreme >= best_success_rate[2] and success_rate_dlp >= best_success_rate[3] and i>100:
            raw_best_success_rate = np.array([success_rate_normal, success_rate_complex, success_rate_extreme, success_rate_dlp])
            best_success_rate = list(np.minimum(raw_best_success_rate, scene_chooser.target_success_rate))
            parking_agent.save("%s/PPO_best.pt" % (save_path),params_only=True)
            f_best_log = open(save_path+'best.txt', 'w')
            f_best_log.write('epoch: %s, success rate: %s %s %s %s'%(i+1, raw_best_success_rate[0],
                                raw_best_success_rate[1], raw_best_success_rate[2], raw_best_success_rate[3]))
            f_best_log.close()
        if (i+1) % 2000 == 0:
            parking_agent.save("%s/PPO2_%s.pt" % (save_path, i),params_only=True)

        if verbose and i%20==0:
            episodes = [j for j in range(len(reward_list))]
            mean_reward = [np.mean(reward_list[max(0,j-50):j+1]) for j in range(len(reward_list))]
            plt.plot(episodes,reward_list)
            plt.plot(episodes,mean_reward)
            plt.xlabel('episodes')
            plt.ylabel('reward')
            f = plt.gcf()
            f.savefig('%s/reward.png'%save_path)
            f.clear()

    # evaluation
    eval_episode = args.eval_episode
    choose_action = True
    with torch.no_grad():
        # eval on dlp
        env.set_level('dlp')
        log_path = save_path+'/dlp'
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        eval(env, parking_agent, episode=eval_episode, log_path=log_path, post_proc_action=choose_action)
        
        # eval on extreme
        env.set_level('Extrem')
        log_path = save_path+'/extreme'
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        eval(env, parking_agent, episode=eval_episode, log_path=log_path, post_proc_action=choose_action)
        
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