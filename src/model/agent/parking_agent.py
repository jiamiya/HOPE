
class RsPlanner(object):
    def __init__(self, step_ratio:float) -> None:
        self.route = None
        self.actions = []
        self.step_ratio = step_ratio

    def reset(self,):
        self.route = None
        self.actions.clear()
    
    def set_rs_path(self, rs_path):
        action_type = {'L':1, 'S':0, 'R':-1}
        self.route = rs_path
        step_ratio = self.step_ratio
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
        
        self.actions = filtered_actions

    def get_action(self, ):
        action = self.actions.pop(0)
        if len(self.actions) == 0 and self.route is not None:
            self.reset()
        return action

class ParkingAgent(object):
    def __init__(
        self, rl_agent, planner=None,
    ) -> None:
        self.agent = rl_agent
        self.planner = planner

    def __getattr__(self, name: str):
        if name.startswith('_'):
            raise AttributeError("attempted to get missing private attribute '{}'".format(name))
        return getattr(self.agent, name)
    
    def reset(self,):
        if self.planner is not None:
            self.planner.reset()

    def set_planner_path(self, path=None, forced=False):
        if self.planner is None:
            return
        if path is not None and (forced or self.planner.route is None):
            self.planner.set_rs_path(path)

    @property
    def executing_rs(self,):
        return not (self.planner is None or self.planner.route is None)
    
    def get_log_prob(self, obs, action):
        return self.agent.get_log_prob(obs, action)

    def choose_action(self, obs):
        '''
        Get the fused decision from the planner and the agent.
        The action is clipped to the range of the safe action space using action mask.

        Params:
            obs(dict): the observation of the environment

        Return:
            action(np.array): the fused decision
            other: the other information, such as the log_prob of the action in case of PPO
        '''
        if not self.executing_rs:
            return self.agent.choose_action(obs)
        else:
            action = self.planner.get_action()
            log_prob = self.agent.get_log_prob(obs, action)
            return action, log_prob
        
    def get_action(self, obs):
        '''
        Get the fused decision from the planner and the agent.

        Params:
            obs(dict): the observation of the environment

        Return:
            action(np.array): the fused decision
            other: the other information, such as the log_prob of the action in case of PPO
        '''
        if not self.executing_rs:
            return self.agent.get_action(obs)
        else:
            action = self.planner.get_action()
            log_prob = self.agent.get_log_prob(obs, action)
            return action, log_prob
            
    def push_memory(self, experience):
        self.agent.push_memory(experience)

    def update(self,):
        return self.agent.update()
    
    def save(self, *args, **kwargs ):
        self.agent.save(*args, **kwargs )

    def load(self, *args, **kwargs ):
        self.agent.load(*args, **kwargs)