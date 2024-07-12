from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal, Beta
import numpy as np

from model.agent_base import ConfigBase, AgentBase
from model.network import *
from model.replay_memory import ReplayMemory
from model.state_norm import StateNorm
from model.action_mask import ActionMask


class PPOConfig(ConfigBase):
    def __init__(self, configs):
        super().__init__()

        # hyperparameters
        self.lr_actor = self.lr
        self.lr_critic = self.lr*5
        self.adam_epsilon = 1e-8
        self.dist_type = "gaussian"
        self.hidden_size = 256
        self.mini_epoch = 10
        self.mini_batch = 32

        self.clip_epsilon = 0.2
        self.lambda_ = 0.95
        self.var_max = 1

        # tricks
        self.adv_norm = True
        self.state_norm = True
        self.reward_norm = False
        self.use_gae = True
        self.reward_scaling = False
        self.gradient_clip = False
        self.policy_entropy = False
        self.entropy_coef = 0.01

        self.merge_configs(configs)


class PPOAgent(AgentBase):
    def __init__(
        self, configs: dict, discrete: bool = False, verbose: bool = False,
        save_params: bool = False, load_params: bool = False
    ) -> None:

        super().__init__(PPOConfig, configs, verbose, save_params, load_params)
        self.discrete = discrete
        self.action_filter = ActionMask()

        # debug
        self.actor_loss_list = []
        self.critic_loss_list = []

        # the networks
        self._init_network()

        # As a on-policy RL algorithm, PPO does not have memory, the self.memory represents
        # the buffer
        self.memory = ReplayMemory(self.configs.batch_size, ["log_prob","next_obs"])

        # tricks
        if self.configs.state_norm:
            self.state_normalize = StateNorm(self.configs.observation_shape)

        
    def _init_network(self):
        '''
        Initialize 1.the network, 2.the optimizer, 3.the checklist.
        '''
        self.actor_net = \
            MultiObsEmbedding(self.configs.actor_layers).to(self.device)
        if self.configs.dist_type == "gaussian":
            self.log_std = \
                nn.Parameter(
                    torch.zeros(1, self.configs.action_dim), requires_grad=False
                ).to(self.device)
            self.log_std.requires_grad = True
            self.actor_optimizer = \
                torch.optim.Adam(
                    [{'params':self.actor_net.parameters()}, {'params': self.log_std}], 
                    self.configs.lr_actor, 
                    # eps=self.configs.lr_actor
                )
        else:
            self.actor_optimizer = \
                torch.optim.Adam(
                    self.actor_net.parameters(), 
                    eps=self.configs.lr_actor
                )

        self.critic_net = \
            MultiObsEmbedding(self.configs.critic_layers).to(self.device)
        self.critic_optimizer = \
            torch.optim.Adam(
                self.critic_net.parameters(), 
                self.configs.lr_critic,
                eps=self.configs.adam_epsilon
            )
        self.critic_target = deepcopy(self.critic_net).to(self.device)
        
        # save and load
        self.check_list = [ # (name, item, save_state_dict)
            ("configs", self.configs, 0),
            ("actor_net", self.actor_net, 1),
            ("actor_optimizer", self.actor_optimizer, 1),
            ("critic_net", self.critic_net, 1),
            ("critic_optimizer", self.critic_optimizer, 1),
            ("critic_target", self.critic_target, 1)
        ]
        if self.configs.dist_type == "gaussian":
            self.check_list.append(("log_std", self.log_std, 0))

    def _actor_forward(self, obs) -> torch.distributions.Distribution: # to be replaced
        observation = deepcopy(obs)
        if self.configs.state_norm:
            observation = self.state_normalize.state_norm(observation)
        observation = self.obs2tensor(observation)
        
        with torch.no_grad():
            policy_dist = self.actor_net(observation)
            if len(policy_dist.shape) > 1 and policy_dist.shape[0] > 1:
                raise NotImplementedError
            if self.discrete:
                dist = Categorical(F.softmax(policy_dist, dim=1))
            elif self.configs.dist_type == "beta":
                alpha, beta = torch.chunk(policy_dist, 2, dim=-1)
                alpha = F.softplus(alpha) + 1.0
                beta = F.softplus(beta) + 1.0
                dist = Beta(alpha, beta)
            elif self.configs.dist_type == "gaussian":
                mean =  torch.clamp(policy_dist,-1,1)  
                log_std = self.log_std.expand_as(mean)  # To make 'log_std' have the same dimension as 'mean'
                std = torch.exp(log_std)
                dist = Normal(mean, std)
            else:
                raise NotImplementedError
            
        return dist
    
    def _post_process_action(self, action_dist:torch.distributions.Distribution , action_mask=None): # to be replaced
        if action_mask is not None:
            mean, std = action_dist.mean, action_dist.stddev
            action = self.action_filter.choose_action(mean, std, action_mask)
            action = torch.FloatTensor(action).to(self.device)
        else:
            action = action_dist.sample()

        if not self.discrete and self.configs.dist_type == "gaussian":
                action = torch.clamp(action, -1, 1)
        log_prob = action_dist.log_prob(action)
        action = action.detach().cpu().numpy().flatten()
        log_prob = log_prob.detach().cpu().numpy().flatten()
        return action, log_prob


    def choose_action(self, obs):

        dist = self._actor_forward(obs)
        action_mask = obs['action_mask']
        action, other_info = self._post_process_action(dist, action_mask)
                
        return action, other_info

    def get_action(self, obs: np.ndarray):
        '''Take action based on one observation. 

        Args:
            observation(np.ndarray): np.ndarray with the same shape of self.state_dim.

        Returns:
            action: If self.discrete, the action is an (int) index. 
                If the action space is continuous, the action is an (np.ndarray).
            log_prob(np.ndarray): the log probability of taken action.
        '''
        dist = self._actor_forward(obs)
        action, log_prob = self._post_process_action(dist)
                
        return action, log_prob

    def get_log_prob(self, obs: np.ndarray, action: np.ndarray):
        '''get the log probability for given action based on current policy

        Args:
            observation(np.ndarray): np.ndarray with the same shape of self.state_dim.

        Returns:
            log_prob(np.ndarray): the log probability of taken action.
        '''
        dist = self._actor_forward(obs)
        
        action = torch.FloatTensor(action).to(self.device)
        log_prob = dist.log_prob(action)
        log_prob = log_prob.detach().cpu().numpy().flatten()
        return log_prob

    def push_memory(self, observations):
        '''
        Args:
            observations(tuple): (obs, action, reward, done, log_prob, next_obs)
        '''
        obs, action, reward, done, log_prob, next_obs = deepcopy(observations)
        if self.configs.state_norm:
            obs = self.state_normalize.state_norm(obs)
            next_obs = self.state_normalize.state_norm(next_obs,update=True)
        observations = (obs, action, reward, done, log_prob, next_obs)
        self.memory.push(observations)

    def _reward_norm(self, reward):
        return (reward - reward.mean()) / (reward.std() + 1e-8)

    def obs2tensor(self, obs):
        if isinstance(obs, list):
            merged_obs = {}
            for obs_type in self.configs.observation_shape.keys():
                merged_obs[obs_type] = []
                for o in obs:
                    merged_obs[obs_type].append(o[obs_type])
                merged_obs[obs_type] = torch.FloatTensor(np.array(merged_obs[obs_type])).to(self.device)
            obs = merged_obs 
        elif isinstance(obs, dict):
            for obs_type in self.configs.observation_shape.keys():
                obs[obs_type] = torch.FloatTensor(obs[obs_type]).to(self.device).unsqueeze(0)
        else:
            raise NotImplementedError()
        return obs
    
    def get_obs(self, obs, ids):
        return {k:obs[k][ids] for k in obs }

    def update(self): # to be replaced
        # convert batches to tensors

        # GAE computation cannot use shuffled data
        # batches = self.memory.shuffle()
        batches = self.memory.get_items(np.arange(len(self.memory)))
        state_batch = self.obs2tensor(batches["state"])
        
        if self.discrete:
            action_batch = torch.IntTensor(batches["action"]).to(self.device)
        else:
            action_batch = torch.FloatTensor(batches["action"]).to(self.device)
        rewards = torch.FloatTensor(np.array(batches["reward"])).unsqueeze(1)
        reward_batch = self._reward_norm(rewards) \
            if self.configs.reward_norm else rewards
        reward_batch = reward_batch.to(self.device)
        done_batch = torch.FloatTensor(batches["done"]).to(self.device).unsqueeze(1)
        old_log_prob_batch = torch.FloatTensor(batches["log_prob"]).to(self.device)
        next_state_batch = self.obs2tensor(batches["next_obs"])
        self.memory.clear()

        # GAE
        gae = 0
        adv = []

        with torch.no_grad():
            value = self.critic_net(state_batch)
            next_value = self.critic_net(next_state_batch)
            deltas = reward_batch + self.configs.gamma * (1 - done_batch) * next_value - value
            if self.configs.use_gae:
                for delta, done in zip(reversed(deltas.cpu().flatten().numpy()), reversed(done_batch.cpu().flatten().numpy())):
                    gae = delta + self.configs.gamma * self.configs.lambda_ * gae * (1.0 - done)
                    adv.append(gae)
                adv.reverse()
                adv = torch.FloatTensor(adv).view(-1, 1).to(self.device)
            else:
                adv = deltas
            v_target = adv + self.critic_target(state_batch)
            if self.configs.adv_norm: # advantage normalization
                adv = (adv - adv.mean()) / (adv.std() + 1e-5)
        
        # apply multi update epoch
        for _ in range(self.configs.mini_epoch):
            # use mini batch and shuffle data
            mini_batch = self.configs.mini_batch
            batchsize = self.configs.batch_size
            train_times = batchsize//mini_batch if batchsize%mini_batch==0 else batchsize//mini_batch+1
            random_idx = np.arange(batchsize)
            np.random.shuffle(random_idx)
            for i in range(train_times):
                if i == batchsize//mini_batch:
                    ri = random_idx[i*mini_batch:]
                else:
                    ri = random_idx[i*mini_batch:(i+1)*mini_batch]
                # state = state_batch[ri]
                state = self.get_obs(state_batch, ri)
                if self.discrete:
                    dist = Categorical(F.softmax(self.actor_net(state),dim=-1))
                    dist_entropy = dist.entropy().view(-1, 1)
                    log_prob= dist.log_prob(action_batch[ri].squeeze()).view(-1, 1)
                    old_log_prob = old_log_prob_batch[ri].view(-1,1)
                elif self.configs.dist_type == "beta":
                    policy_dist = self.actor_net(state)
                    alpha, beta = torch.chunk(policy_dist, 2, dim=-1)
                    alpha = F.softplus(alpha) + 1.0
                    beta = F.softplus(beta) + 1.0
                    dist = Beta(alpha, beta)
                    dist_entropy = dist.entropy().sum(1, keepdim=True)
                    log_prob = dist.log_prob(action_batch[ri])
                    log_prob =torch.sum(log_prob,dim=1, keepdim=True)
                    old_log_prob =torch.sum(old_log_prob_batch[ri],dim=1, keepdim=True)
                elif self.configs.dist_type == "gaussian":
                    policy_dist = self.actor_net(state)
                    mean = torch.clamp(policy_dist, -1, 1)
                    log_std = self.log_std.expand_as(mean)
                    std = torch.exp(log_std)
                    dist = Normal(mean, std)
                    dist_entropy = dist.entropy().sum(1, keepdim=True)
                    log_prob = dist.log_prob(action_batch[ri])
                    log_prob =torch.sum(log_prob,dim=1, keepdim=True)
                    old_log_prob =torch.sum(old_log_prob_batch[ri],dim=1, keepdim=True)
                prob_ratio = (log_prob - old_log_prob).exp()

                loss1 = prob_ratio * adv[ri]
                loss2 = torch.clamp(prob_ratio, 1 - self.configs.clip_epsilon, 1 + self.configs.clip_epsilon) * adv[ri]

                actor_loss = - torch.min(loss1, loss2)
                if self.configs.policy_entropy:
                    actor_loss += - self.configs.entropy_coef * dist_entropy
                critic_loss = F.mse_loss(v_target[ri], self.critic_net(state))

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                actor_loss.mean().backward()
                critic_loss.mean().backward()
                
                self.actor_loss_list.append(actor_loss.mean().item())
                self.critic_loss_list.append(critic_loss.mean().item())
                if self.configs.gradient_clip: # gradient clip
                    nn.utils.clip_grad_norm_(self.critic_net.parameters(), 0.5)
                    nn.utils.clip_grad_norm(self.actor_net.parameters(), 0.5)
                self.actor_optimizer.step()
                self.critic_optimizer.step() 

            self._soft_update(self.critic_target, self.critic_net)

        if self.configs.lr_decay: # learning rate decay
            self.actor_optimizer.param_groups["lr"] = self.lr_decay(self.configs.lr_actor)
            self.critic_optimizer.param_groups["lr"] = self.lr_decay(self.configs.lr_critic)

        # for debug
        a = actor_loss.detach().cpu().numpy()[0][0]
        b = critic_loss.item()
        return a, b

    def save(self, path: str = None, params_only: bool = None) -> None: # to be replaced
        """Store the model structure and corresponding parameters to a file.
        """
        if params_only is not None:
            self.save_params = params_only
        if self.save_params and len(self.check_list) > 0:
            checkpoint = dict()
            for name, item, save_state_dict in self.check_list:
                checkpoint[name] = item.state_dict() if save_state_dict else item
            # for PPO extra save
            if self.configs.dist_type == "gaussian":
                checkpoint['log'] = self.log_std
            checkpoint['state_norm'] = self.state_normalize # (self.state_mean, self.state_std, self.S, self.n_state)
            checkpoint['optimizer'] = (self.actor_optimizer, self.critic_optimizer)
            torch.save(checkpoint, path)
        else:
            torch.save(self, path)
        
        if self.verbose:
            print("Save current model to %s" % path)

    def load(self, path: str = None, params_only: bool = None) -> None: # to be replaced
        """Load the model structure and corresponding parameters from a file.
        """
        if params_only is not None:
            self.load_params = params_only
        if self.load_params and len(self.check_list) > 0:
            checkpoint = torch.load(path, map_location=self.device)
            for name, item, save_state_dict in self.check_list:
                if save_state_dict:
                    item.load_state_dict(checkpoint[name])
                else:
                    item = checkpoint[name]

            self.log_std.data.copy_(checkpoint['log']) 
            
            self.state_normalize = checkpoint['state_norm'] 
            if 'optimizer' in checkpoint.keys():
                self.actor_optimizer, self.critic_optimizer = checkpoint['optimizer']
        else:
            torch.load(self, path)
        
            path =f"{path}/{name}_{id}.pth"
            state_dict = torch.load(path, map_location=self.device)
            object.load_state_dict(state_dict)

        if self.verbose:
            print("Load the model from %s" % path)

    def load_actor(self, path: str = None) -> None: # to be replaced
        """Load the model structure and corresponding parameters from a file.
        """
        if len(self.check_list) > 0:
            checkpoint = torch.load(path, map_location=self.device)
            for name, item, save_state_dict in self.check_list:
                if name != 'actor_net':
                    continue
                if save_state_dict:
                    item.load_state_dict(checkpoint[name])
                else:
                    item = checkpoint[name]

            self.log_std.data.copy_(checkpoint['log']) 
            # self.actor_target_net = deepcopy(self.actor_net).to(self.device)
            self.state_normalize = checkpoint['state_norm']

    def load_img_encoder(self, path: str = None, require_grad: bool = False) -> None:
        self.actor_net.load_img_encoder(path, self.device, require_grad)
        self.critic_net.load_img_encoder(path, self.device, require_grad)
        self.critic_target = deepcopy(self.critic_net).to(self.device)
        print('Load pretrained image encoder from path: %s'%path)