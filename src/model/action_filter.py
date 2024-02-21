import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage.filters import minimum_filter1d
from scipy.interpolate import interpn, LinearNDInterpolator

from configs import *


class ActionFilter(nn.Module):
    def __init__(self, configs=CONFIGS_ACTION):
        super().__init__()
        activate_func = [nn.LeakyReLU(), nn.Tanh()][configs['use_tanh_activate']]
        hidden_size = configs['hidden_size']
        layers = [nn.Linear(configs['lidar_shape'], hidden_size), activate_func]
        for _ in range(configs['n_hidden_layers']-2):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(activate_func)
        layers.extend([nn.Linear(hidden_size, configs['n_action']), nn.Sigmoid()])
        self.net = nn.Sequential(*layers)
        self.action_space = configs['discrete_actions']

    def forward(self, lidar_view):
        if isinstance(lidar_view, np.ndarray):
            lidar_view = torch.FloatTensor(lidar_view).to(device)
        if len(lidar_view.shape) == 1:
            lidar_view.unsqueeze(0)
        action_confidence = self.net(lidar_view)
        return action_confidence
    
    def load(self, path):
        model = torch.load(path)
        self.net = model.net

    def choose_action(self, action_mean, action_std, action_mask):

        if isinstance(action_mean, torch.Tensor):
            action_mean = action_mean.cpu().numpy()
            action_std = action_std.cpu().numpy()
        if isinstance(action_mask, torch.Tensor):
            action_mask = action_mask.cpu().numpy()
        if len(action_mean.shape) == 2:
            action_mean = action_mean.squeeze(0)
            action_std = action_std.squeeze(0)
        if len(action_mask.shape) == 2:
            action_mask = action_mask.squeeze(0)

        def calculate_probability(mean, std, values):
            z_scores = (values - mean) / std
            log_probabilities = -0.5 * z_scores ** 2 - np.log((np.sqrt(2 * np.pi) * std))
            # print(log_probabilities)
            return np.sum(np.clip(log_probabilities, -10, 10), axis=1)
        possible_actions = np.array(self.action_space)
        # deal the scaling
        action_mean[1] = 1 if action_mean[1]>0 else -1 # TODO
        scale_steer = VALID_STEER[1]
        scale_speed = 1
        possible_actions = possible_actions/np.array([scale_steer, scale_speed])
        # print(possible_actions)
        prob = calculate_probability(action_mean, action_std, possible_actions)
        # prob = np.clip(prob, -10, 10)
        # prob -= prob.min()
        exp_prob = np.exp(prob) * action_mask
        prob_softmax = exp_prob / np.sum(exp_prob)
        # print(prob)
        # print(np.round(prob_softmax, 3))
        actions = np.arange(len(possible_actions))
        action_chosen = np.random.choice(actions, p=prob_softmax)
        # action_chosen = np.argmax(prob_softmax)
        # print(action_chosen, action_chosen1)
        # print(action_chosen)
        # return action_mean
        # possible_actions[action_chosen][1] = action_mean[1]
        return possible_actions[action_chosen]
    

class ActionFilter2():
    def __init__(self, VehicleBox=VehicleBox, lidar_base=vehicle_base, n_iter=10) -> None:
        self.vehicle_box_base = VehicleBox
        self.n_iter = n_iter
        self.action_space = discrete_actions
        self.vehicle_boxes = self.init_vehicle_box()
        self.vehicle_lidar_base = lidar_base
    
    def init_vehicle_box(self,):
        VehicleBox = self.vehicle_box_base
        vehicle_boxes = []
        x,y,theta = 0,0,0#vehicle_state.loc.x, vehicle_state.loc.y, vehicle_state.heading
        actions = np.array(self.action_space)
        radius = 1/(np.tan(actions[:,0])/WHEEL_BASE) # TODO: the steer is 0
        car_coords = np.array(VehicleBox.coords)[:4] # (4,2)
        # print(car_coords)
        # car_coords[car_coords>0] += 0.1
        # car_coords[car_coords<0] -= 0.1
        # print(car_coords)
        # p
        car_coords_x = car_coords[:,0].reshape(1,-1)
        car_coords_y = car_coords[:,1].reshape(1,-1) # (1,4)
        # print(car_coords)
        # print(radius)
        Ox = x-radius*np.sin(theta)
        Oy = y+radius*np.cos(theta)
        delta_phi = 0.5*actions[:,1]/10/radius # (42) TODO: simu time
        ptheta = theta
        px, py = x,y

        for i in range(self.n_iter):
            ptheta = ptheta + delta_phi # (42)
            px = Ox + radius*np.sin(ptheta) # (42)
            py = Oy - radius*np.cos(ptheta) # (42)

            # coords transform
            cos_theta = np.cos(ptheta).reshape(-1,1) # (42)
            sin_theta = np.sin(ptheta).reshape(-1,1)
            vehicle_coords_x = cos_theta*car_coords_x - sin_theta*car_coords_y + px.reshape(-1,1) # (42,4)
            vehicle_coords_y = sin_theta*car_coords_x + cos_theta*car_coords_y + py.reshape(-1,1) # (42,4)
            vehicle_coords = np.concatenate((np.expand_dims(vehicle_coords_x, axis=-1), np.expand_dims(vehicle_coords_y, axis=-1)), axis=-1) # (42,4,2)
            # mat = [cos_theta, -sin_theta, sin_theta, cos_theta, self.loc.x, self.loc.y]
            
            # vehicle_coords_ = vehicle_coords.reshape((vehicle_coords.shape[0], 1, vehicle_coords.shape[1], vehicle_coords.shape[2])) # (42,1,4,2)
            vehicle_boxes.append(vehicle_coords)
        
        return vehicle_boxes
    
    def _linear_interpolate(self, x:np.ndarray, upsample_rate:int=None):
        '''
        Interpolate the input array by linear interpolation on the first axis.

        Parameters:
            `x`: the input array
            `upsample_rate`: the upsample rate
        Returns:
            `y`: y[j,...] = x[j//upsample_rate,...]*((j)%upsample_rate)/upsample_rate + x[(j)//upsample_rate+1,...]*(1-(j%upsample_rate)/upsample_rate)
        '''
        if upsample_rate is None:
            upsample_rate = 2
        y = np.zeros((x.shape[0]*upsample_rate,) + x.shape[1:])
        x = np.concatenate([x, x[0:1]], axis=0) # assume the input is circular
        j = np.arange(y.shape[0])
        tmp_shape = (y.shape[0],) + (1,)*(len(x.shape)-1)
        y = x[j//upsample_rate,...]*(1-(j%upsample_rate)/upsample_rate).reshape(tmp_shape) +\
            x[(j)//upsample_rate+1,...]*(((j)%upsample_rate)/upsample_rate).reshape(tmp_shape)
        return y


    def get_steps(self, raw_lidar_obs:np.ndarray):
        '''
        raw_lidar_obs: the raw lidar obs which already substract the vehicle base.
        '''
        lidar_obs = np.clip(raw_lidar_obs, 0, 10) + self.vehicle_lidar_base
        # lidar_obs = self._linear_interpolate(lidar_obs, upsample_rate=5) # (240,)
        lidar_num = len(lidar_obs)
        angle_vec = np.arange(lidar_num)*np.pi/lidar_num*2
        obstacle_range_x = np.cos(angle_vec)*lidar_obs
        obstacle_range_y = np.sin(angle_vec)*lidar_obs
        obstacle_range_coords = np.concatenate(
            (np.expand_dims(obstacle_range_x, 1), np.expand_dims(obstacle_range_y, 1)), axis=1) # (120, 2)
        
        """ 
        obstacle_range_x_out = np.cos(angle_vec)*(lidar_obs+1)
        obstacle_range_y_out = np.sin(angle_vec)*(lidar_obs+1)
        obstacle_range_coords_out = np.concatenate(
            (np.expand_dims(obstacle_range_x_out, 1), np.expand_dims(obstacle_range_y_out, 1)), axis=1) # (120, 2)
        lidar_obst_a = obstacle_range_coords[:-1] # (119, 2)
        lidar_obst_b = obstacle_range_coords[1:]
        lidar_obst_c = obstacle_range_coords_out[1:]
        lidar_obst_d = obstacle_range_coords_out[:-1]
        lidar_obstacles = np.concatenate((lidar_obst_a.reshape(-1,1,2),
                            lidar_obst_b.reshape(-1,1,2),
                            lidar_obst_c.reshape(-1,1,2),
                            lidar_obst_d.reshape(-1,1,2),), axis=1) # (119, 4, 2)
        lidar_obst_area = 0.5 * np.abs(np.cross(lidar_obst_a-lidar_obst_b, lidar_obst_a-lidar_obst_d))\
            + 0.5 * np.abs(np.cross(lidar_obst_c-lidar_obst_b, lidar_obst_c-lidar_obst_d)) # (119)
        lidar_obstacles_ = lidar_obstacles.reshape((1,1,)+lidar_obstacles.shape)  # (1, 1, 119, 4, 2)
    """
        shifted_obstacle_coords = obstacle_range_coords.copy()
        shifted_obstacle_coords[:-1] = obstacle_range_coords[1:]
        shifted_obstacle_coords[-1] = obstacle_range_coords[0]
        obstacle_edge_vec = shifted_obstacle_coords - obstacle_range_coords # (120, 2)
        # # plt.show()
        
        # x,y,theta = 0,0,0#vehicle_state.loc.x, vehicle_state.loc.y, vehicle_state.heading
        # actions = np.array(discrete_actions)
        # radius = 1/(np.tan(actions[:,0])/WHEEL_BASE) # TODO: the steer is 0
        # car_coords = np.array(VehicleBox.coords)[:4] # (4,2)
        # car_coords_x = car_coords[:,0].reshape(1,-1)
        # car_coords_y = car_coords[:,1].reshape(1,-1) # (1,4)
        vehicle_area = LENGTH*WIDTH
        # Ox = x-radius*np.sin(theta)
        # Oy = y+radius*np.cos(theta)
        # delta_phi = 0.5*actions[:,1]/10/radius # (42) TODO: simu time
        # ptheta = theta
        # px, py = x,y

        collides = []
        for i in range(self.n_iter):
            # ptheta = ptheta + delta_phi # (42)
            # px = Ox + radius*np.sin(ptheta) # (42)
            # py = Oy - radius*np.cos(ptheta) # (42)

            # # coords transform
            # cos_theta = np.cos(ptheta).reshape(-1,1) # (42)
            # sin_theta = np.sin(ptheta).reshape(-1,1)
            # vehicle_coords_x = cos_theta*car_coords_x - sin_theta*car_coords_y + px.reshape(-1,1) # (42,4)
            # vehicle_coords_y = sin_theta*car_coords_x + cos_theta*car_coords_y + py.reshape(-1,1) # (42,4)
            # vehicle_coords = np.concatenate((np.expand_dims(vehicle_coords_x, axis=-1), np.expand_dims(vehicle_coords_y, axis=-1)), axis=-1) # (42,4,2)
            vehicle_coords = self.vehicle_boxes[i]
            # mat = [cos_theta, -sin_theta, sin_theta, cos_theta, self.loc.x, self.loc.y]
            
            
            obstacle_range_coords_ = obstacle_range_coords.reshape((1, obstacle_range_coords.shape[0], 1, obstacle_range_coords.shape[1])) # (1,120,1,2)
            vehicle_coords_ = vehicle_coords.reshape((vehicle_coords.shape[0], 1, vehicle_coords.shape[1], vehicle_coords.shape[2])) # (42,1,4,2)
            vec1 = obstacle_range_coords_ - vehicle_coords_ # (42,120,4,2). (4,2) -> [(L-a), (L-b), (L-c), (L-d)]
            # print(vec1.shape)
            # print(np.cross(vec1[:,:,0,:], vec1[:,:,1,:]).shape)
            area1 = 0.5 * np.abs(np.cross(vec1[:,:,0,:], vec1[:,:,1,:])) # (42, 120)
            area2 = 0.5 * np.abs(np.cross(vec1[:,:,1,:], vec1[:,:,2,:]))
            area3 = 0.5 * np.abs(np.cross(vec1[:,:,2,:], vec1[:,:,3,:]))
            area4 = 0.5 * np.abs(np.cross(vec1[:,:,3,:], vec1[:,:,0,:]))
            area = area1 + area2 + area3 + area4 # (42, 120)

            collide1 = np.sum((area < vehicle_area+1e-8).astype(int), axis=1) # (42,)
            # print(collide)
            # collide1 = (collide1!=0).astype(int)
            
            # judge the vehicle point intersects the lidar obstacle
            """ 
            t = time.time()
            vehicle_coords_2 = vehicle_coords.reshape(vehicle_coords.shape[:2]+(1,1,2)) # (42,4,1,1,2)
            vec2 = vehicle_coords_2 - lidar_obstacles_  # (42,4,119,4,2)
            area1 = 0.5 * np.abs(np.cross(vec2[:,:,:,0,:], vec2[:,:,:,1,:])) # (42,4,119)
            area2 = 0.5 * np.abs(np.cross(vec2[:,:,:,1,:], vec2[:,:,:,2,:]))
            area3 = 0.5 * np.abs(np.cross(vec2[:,:,:,2,:], vec2[:,:,:,3,:]))
            area4 = 0.5 * np.abs(np.cross(vec2[:,:,:,3,:], vec2[:,:,:,0,:]))
            area = area1 + area2 + area3 + area4 # (42, 4, 119)
            collide2 = np.sum((area < lidar_obst_area+1e-8).astype(int), axis=(1,2))
            print(collide2)
            collide2 = (collide2!=0).astype(int)
            print('*',time.time()-t)
            """
            cross1 = np.cross(vec1[:,:,0,:], obstacle_edge_vec) # (42, 120)
            cross2 = np.cross(vec1[:,:,1,:], obstacle_edge_vec) # (42, 120)
            cross3 = np.cross(vec1[:,:,2,:], obstacle_edge_vec) # (42, 120)
            cross4 = np.cross(vec1[:,:,3,:], obstacle_edge_vec) # (42, 120)
            # print(cross1*cross3)
            # collide2_1 = np.sum(((cross1*cross3 < 0) + (cross2*cross4 < 0)).astype(int), axis=-1) # (42,)
            collide2_1 = ((cross1*cross3 < 0) * (cross2*cross4 < 0)).astype(int) # (42,120)

            vehicle_diag_vec1 = vehicle_coords[:,0:1,:] - vehicle_coords[:,2:3,:] # (42,1,2)
            vehicle_diag_vec2 = vehicle_coords[:,1:2,:] - vehicle_coords[:,3:4,:]
            vec1_shifted = vec1.copy() # (42,120,4,2)
            vec1_shifted[:,:-1] = vec1[:,1:]
            vec1_shifted[:,-1] = vec1[:,0]
            cross5 = np.cross(vec1[:,:,0,:], vehicle_diag_vec1)
            cross6 = np.cross(vec1_shifted[:,:,0,:], vehicle_diag_vec1)
            cross7 = np.cross(vec1[:,:,1,:], vehicle_diag_vec2)
            cross8 = np.cross(vec1_shifted[:,:,1,:], vehicle_diag_vec2)
            # collide2_2 = np.sum(((cross5*cross6 < 0) + (cross7*cross8 < 0)).astype(int), axis=-1) # (42,)
            collide2_2 = ((cross5*cross6 < 0) * (cross7*cross8 < 0)).astype(int) # (42,120)
            collide2 = np.sum(collide2_1*collide2_2, axis=-1)
            # print("#%s "%i,collide2)
            # print((collide2_1*collide2_2)[0])
            # idx = np.argmax((collide2_1*collide2_2)[-5])
            # print((cross5*cross6)[0,idx:], (cross7*cross8)[0,idx:])
            # if i <= 2:
            #     # print(vehicle_coords_x,vehicle_coords_y)
            #     # from shapely.affinity import affine_transform
            #     # print(affine_transform())
            #     # plt.scatter(vehicle_coords_x,vehicle_coords_y)
            #     import matplotlib.pyplot as plt
            #     print(i)
            #     fig=plt.figure()
            #     ax=fig.add_subplot(111)
            #     # ax.set_xlim(-10,10)
            #     # ax.set_ylim(-10,10)
            #     # ax.set_xlabel('x')
            #     for i in range(len(lidar_obs)):
            #         # line=plt.Line2D((0,0),(math.cos(i*math.pi/180)*lidar_view[i],math.sin(i*math.pi/180)*lidar_view[i]))
            #         ax.add_line(plt.Line2D((0,np.cos(i*np.pi/lidar_num*2)*lidar_obs[i]), (0,np.sin(i*np.pi/lidar_num*2)*lidar_obs[i])))
            #         if i == idx or i == idx+1:
            #             ax.add_line(plt.Line2D((0,np.cos(i*np.pi/lidar_num*2)*lidar_obs[i]), (0,np.sin(i*np.pi/lidar_num*2)*lidar_obs[i]), color='green'))
            #     ax.add_patch(plt.Polygon(xy=vehicle_coords[-5], color='red'))
            #     plt.xlim(-5,5)
            #     plt.ylim(-5,5)
            #     plt.show()
            # collide2 = 0
            collide = (collide1+collide2 != 0).astype(int)

            collides.append(collide)
        
        collides = np.array(collides) # (10, 42)
        collide_free_binary  = (np.sum(collides, axis=0)==0) # (42)
        step_len = np.argmax(collides, axis=0)
        step_len[collide_free_binary.astype(bool)] = self.n_iter
        # print('raw:\n',step_len)
        step_len = self.post_process(step_len)
        if np.sum(step_len) == 0:
            return np.clip(step_len, 0.01, 1)
            print("###")
            # print(step_len)
            # print(collide_free_binary)
            # print(collides)
            print(lidar_obs)
            print(lidar_obs-vehicle_base)
            import matplotlib.pyplot as plt
            fig=plt.figure()
            ax=fig.add_subplot(111)
            # ax.set_xlim(-10,10)
            # ax.set_ylim(-10,10)
            # ax.set_xlabel('x')
            for i in range(len(lidar_obs)):
                # line=plt.Line2D((0,0),(math.cos(i*math.pi/180)*lidar_view[i],math.sin(i*math.pi/180)*lidar_view[i]))
                ax.add_line(plt.Line2D((0,np.cos(i*np.pi/lidar_num*2)*lidar_obs[i]), (0,np.sin(i*np.pi/lidar_num*2)*lidar_obs[i])))
            ax.add_patch(plt.Polygon(xy=np.array(VehicleBox.coords)[:4], color='red'))
            plt.xlim(-10,10)
            plt.ylim(-10,10)
            plt.show()
            p
        return step_len
    
    def post_process(self, step_len:np.ndarray):
        kernel = 5
        forward_step_len = step_len[:len(step_len)//2]
        backward_step_len = step_len[len(step_len)//2:]
        forward_step_len[0] -= 1
        forward_step_len[-1] -= 1
        backward_step_len[0] -= 1
        backward_step_len[-1] -= 1
        forward_step_len_ = minimum_filter1d(forward_step_len, kernel)
        backward_step_len_ = minimum_filter1d(backward_step_len, kernel)
        return np.clip(np.concatenate((forward_step_len_, backward_step_len_)),0,10)/10
    
    
    def choose_action(self, action_mean, action_std, action_mask):

        if isinstance(action_mean, torch.Tensor):
            action_mean = action_mean.cpu().numpy()
            action_std = action_std.cpu().numpy()
        if isinstance(action_mask, torch.Tensor):
            action_mask = action_mask.cpu().numpy()
        if len(action_mean.shape) == 2:
            action_mean = action_mean.squeeze(0)
            action_std = action_std.squeeze(0)
        if len(action_mask.shape) == 2:
            action_mask = action_mask.squeeze(0)

        def calculate_probability(mean, std, values):
            z_scores = (values - mean) / std
            log_probabilities = -0.5 * z_scores ** 2 - np.log((np.sqrt(2 * np.pi) * std))
            # print(log_probabilities)
            return np.sum(np.clip(log_probabilities, -10, 10), axis=1)
        possible_actions = np.array(self.action_space)
        # deal the scaling
        action_mean[1] = 1 if action_mean[1]>0 else -1 # TODO
        scale_steer = VALID_STEER[1]
        scale_speed = 1
        possible_actions = possible_actions/np.array([scale_steer, scale_speed])
        # print(possible_actions)
        prob = calculate_probability(action_mean, action_std, possible_actions)
        # prob = np.clip(prob, -10, 10)
        # prob -= prob.min()
        exp_prob = np.exp(prob) * action_mask
        prob_softmax = exp_prob / np.sum(exp_prob)
        # print(prob)
        # print(np.round(prob_softmax, 3))
        actions = np.arange(len(possible_actions))
        action_chosen = np.random.choice(actions, p=prob_softmax)
        # action_chosen = np.argmax(prob_softmax)
        # possible_actions[action_chosen][1] = action_mean[1]
        return possible_actions[action_chosen]
        

class ActionMask():
    def __init__(self, VehicleBox=VehicleBox, lidar_base=vehicle_base, n_iter=10) -> None:
        self.vehicle_box_base = VehicleBox
        self.n_iter = n_iter
        self.action_space = discrete_actions
        self.vehicle_boxes = self.init_vehicle_box() #(42, n_iter, 4, 2)
        self.vehicle_lidar_base = lidar_base
        self.lidar_num = LIDAR_NUM
        self.up_sample_rate = 10
        self.dist_star = self.precompute()

    def _intersect(self, e1:np.ndarray, e2:np.ndarray):
        """
        Calcualte the intersection of 2 group of edges.

        Params:
            e1 (array): coordinates of m edges in shape (m*2*2)
            e2 (array): coordinates of n edges in shape (n*2*2)

        Return:
            intersections (array): coordinates of m*n intersections in shape (m*n*2).
            If 2 edges do not intersects, the corresponding value is set as +np.inf.
        """
        # calculate the params for edges 1
        x1s1, x2s1, y1s1, y2s1 = e1[:, 0, 0].reshape(-1,1), e1[:, 1, 0].reshape(-1,1), e1[:, 0, 1].reshape(-1,1), e1[:, 1, 1].reshape(-1,1)
        a = (y2s1 - y1s1).reshape(-1,1) # (m,1)
        b = (x1s1 - x2s1).reshape(-1,1)
        c = (y1s1*x2s1 - x1s1*y2s1).reshape(-1,1)

        # calculate the params for edges 2
        x1s2, x2s2, y1s2, y2s2 = e2[:, 0, 0].reshape(1,-1), e2[:, 1, 0].reshape(1,-1), e2[:, 0, 1].reshape(1,-1), e2[:, 1, 1].reshape(1,-1)
        d = (y2s2 - y1s2).reshape(1,-1) # (1,n)
        e = (x1s2 - x2s2).reshape(1,-1)
        f = (y1s2*x2s2 - x1s2*y2s2).reshape(1,-1)

        # calculate the intersections by line(instead of edge) 1 & 2
        det = a*e - b*d # (m, n)
        parallel_line_pos = (det==0) # (m, n)
        det[parallel_line_pos] = 1 # temporarily set "1" to avoid "divided by zero"
        raw_x = (b*f - c*e)/det # (m, n)
        raw_y = (c*d - a*f)/det

        # select the true intersections, set the false positive interesections to inf
        tmp_inf = np.inf
        tolerrance = 1e-8
        # the false positive intersections on line L1(not on ray L1)
        raw_x[raw_x>np.maximum(x1s1, x2s1)+tolerrance] = tmp_inf
        raw_x[raw_x<np.minimum(x1s1, x2s1)-tolerrance] = tmp_inf
        raw_y[raw_y>np.maximum(y1s1, y2s1)+tolerrance] = tmp_inf
        raw_y[raw_y<np.minimum(y1s1, y2s1)-tolerrance] = tmp_inf
        # the false positive intersections on line L2(not on edge L2)
        raw_x[raw_x>np.maximum(x1s2, x2s2)+tolerrance] = tmp_inf
        raw_x[raw_x<np.minimum(x1s2, x2s2)-tolerrance] = tmp_inf
        raw_y[raw_y>np.maximum(y1s2, y2s2)+tolerrance] = tmp_inf
        raw_y[raw_y<np.minimum(y1s2, y2s2)-tolerrance] = tmp_inf
        # the (L1, L2) which are parallel
        raw_x[parallel_line_pos] = tmp_inf

        intersections = np.stack([raw_x, raw_y], axis=2) # (m, n, 2)
        print(e1.shape, e2.shape, intersections.shape)
        assert intersections.shape ==  (e1.shape[0], e2.shape[0], 2)

        return intersections
    
    def init_vehicle_box(self,):
        VehicleBox = self.vehicle_box_base
        vehicle_boxes = []
        x,y,theta = 0,0,0#vehicle_state.loc.x, vehicle_state.loc.y, vehicle_state.heading
        actions = np.array(self.action_space)
        radius = 1/(np.tan(actions[:,0])/WHEEL_BASE) # TODO: the steer is 0
        car_coords = np.array(VehicleBox.coords)[:4] # (4,2)
        car_coords_x = car_coords[:,0].reshape(1,-1)
        car_coords_y = car_coords[:,1].reshape(1,-1) # (1,4)
        Ox = x-radius*np.sin(theta)
        Oy = y+radius*np.cos(theta)
        delta_phi = 0.5*actions[:,1]/10/radius # (42) TODO: simu time
        ptheta = theta
        px, py = x,y

        for _ in range(self.n_iter):
            ptheta = ptheta + delta_phi # (42)
            px = Ox + radius*np.sin(ptheta) # (42)
            py = Oy - radius*np.cos(ptheta) # (42)

            # coords transform
            cos_theta = np.cos(ptheta).reshape(-1,1) # (42)
            sin_theta = np.sin(ptheta).reshape(-1,1)
            vehicle_coords_x = cos_theta*car_coords_x - sin_theta*car_coords_y + px.reshape(-1,1) # (42,4)
            vehicle_coords_y = sin_theta*car_coords_x + cos_theta*car_coords_y + py.reshape(-1,1) # (42,4)
            vehicle_coords = np.concatenate((np.expand_dims(vehicle_coords_x, axis=-1), np.expand_dims(vehicle_coords_y, axis=-1)), axis=-1) # (42,4,2)
            vehicle_boxes.append(vehicle_coords)
        
        return np.array(vehicle_boxes).transpose(1,0,2,3)

    def precompute(self,):
        """
        Precomputation of dist_star, which can accelerate the calculatio of action mask.

        Return:
            dist_star(array): shape (lidar_num, n_action, n_iter). 
            dist_star[i,j,k] stands for the minimal distance of obstacle 
            detected by the i-th lidar line for the j-th action to step k iterations without collision.
        """
        lidar_line_idx = np.arange(self.lidar_num)
        max_distance = LIDAR_RANGE*10
        lidar_line_end_x = np.cos(lidar_line_idx/self.lidar_num*2*np.pi)*max_distance
        lidar_line_end_y = np.sin(lidar_line_idx/self.lidar_num*2*np.pi)*max_distance
        lidar_line_end_coords = np.stack([lidar_line_end_x, lidar_line_end_y], axis=1) # (lidar_num, 2)
        origin = np.zeros_like(lidar_line_end_coords)
        lidar_edges = np.stack([origin, lidar_line_end_coords], axis=1) # (lidar_num, 2, 2)

        vehicle_boxes_shifed = np.zeros_like(self.vehicle_boxes)
        vehicle_boxes_shifed[:, :, :-1, :] = self.vehicle_boxes[:, :, 1:, :]
        vehicle_boxes_shifed[:, :, -1, :] = self.vehicle_boxes[:, :, 0, :]
        vehicle_edges = np.stack([vehicle_boxes_shifed, self.vehicle_boxes],axis=3) # (n_action, n_iter, 4, 2, 2)
        vehicle_edges = vehicle_edges.reshape(-1, 2, 2) # (n_action*n_iter*4, 2, 2)
        intersections = self._intersect(lidar_edges, vehicle_edges) # (lidar_num, n_action*n_iter*4, 2)
        # intersections[intersections==np.inf] = 0
        intersections = intersections.reshape(self.lidar_num, len(self.action_space), self.n_iter, 4, 2)
        intersections = np.linalg.norm(intersections, axis=-1)
        intersections[intersections==np.inf] = 0
        dist_star = np.max(intersections, axis=-1) # (lidar_num, n_action, n_iter)
        dist_star = self._linear_interpolate(dist_star, self.up_sample_rate)
        return dist_star
    
    def _linear_interpolate(self, x:np.ndarray, upsample_rate:int=None):
        '''
        Interpolate the input array by linear interpolation on the first axis.

        Parameters:
            `x`: the input array
            `upsample_rate`: the upsample rate
        Returns:
            `y`: y[j,...] = x[j//upsample_rate,...]*((j)%upsample_rate)/upsample_rate + x[(j)//upsample_rate+1,...]*(1-(j%upsample_rate)/upsample_rate)
        '''
        if upsample_rate is None:
            upsample_rate = self.up_sample_rate
        y = np.zeros((x.shape[0]*upsample_rate,) + x.shape[1:])
        x = np.concatenate([x, x[0:1]], axis=0) # assume the input is circular
        j = np.arange(y.shape[0])
        tmp_shape = (y.shape[0],) + (1,)*(len(x.shape)-1)
        y = x[j//upsample_rate,...]*(1-(j%upsample_rate)/upsample_rate).reshape(tmp_shape) +\
            x[(j)//upsample_rate+1,...]*(((j)%upsample_rate)/upsample_rate).reshape(tmp_shape)
        return y


    def get_steps(self, raw_lidar_obs:np.ndarray):
        '''
        raw_lidar_obs: the raw lidar obs which already substract the vehicle base.
        '''
        lidar_obs = np.clip(raw_lidar_obs, 0, 10) + self.vehicle_lidar_base
        dist_obs = self._linear_interpolate(lidar_obs.reshape(-1), self.up_sample_rate).reshape(-1,1,1) # (lidar_num*upsample_rate, 1, 1)
        step_save = np.zeros_like(self.dist_star) # (lidar_num, n_action, n_iter)
        step_save[self.dist_star<=dist_obs] = 1
        step_save[self.dist_star>dist_obs] = 0
        # find the first "0" on the last axis
        max_step = np.argmin(step_save, axis=-1) # (lidar_num, n_action)
        max_step[np.sum(step_save, axis=-1) == self.n_iter] = self.n_iter
        
        # max_step = np.sum(step_save, axis=-1) # (lidar_num, n_action)
        step_len = np.min(max_step, axis=0) # (n_action)

        step_len = self.post_process(step_len)
        if np.sum(step_len) == 0:
            return np.clip(step_len, 0.01, 1)
        return step_len
    
    def post_process(self, step_len:np.ndarray):
        kernel = 5
        forward_step_len = step_len[:len(step_len)//2]
        backward_step_len = step_len[len(step_len)//2:]
        forward_step_len[0] -= 1
        forward_step_len[-1] -= 1
        backward_step_len[0] -= 1
        backward_step_len[-1] -= 1
        forward_step_len_ = minimum_filter1d(forward_step_len, kernel)
        backward_step_len_ = minimum_filter1d(backward_step_len, kernel)
        return np.clip(np.concatenate((forward_step_len_, backward_step_len_)), 0, self.n_iter)/self.n_iter
    
    
    def choose_action(self, action_mean, action_std, action_mask):

        if isinstance(action_mean, torch.Tensor):
            action_mean = action_mean.cpu().numpy()
            action_std = action_std.cpu().numpy()
        if isinstance(action_mask, torch.Tensor):
            action_mask = action_mask.cpu().numpy()
        if len(action_mean.shape) == 2:
            action_mean = action_mean.squeeze(0)
            action_std = action_std.squeeze(0)
        if len(action_mask.shape) == 2:
            action_mask = action_mask.squeeze(0)
            # action_mask[action_mask <= 0.1] = 0
            # action_mask[action_mask > 0.1] = 1

        def calculate_probability(mean, std, values):
            z_scores = (values - mean) / std
            log_probabilities = -0.5 * z_scores ** 2 - np.log((np.sqrt(2 * np.pi) * std))
            # print(log_probabilities)
            return np.sum(np.clip(log_probabilities, -10, 10), axis=1)
        possible_actions = np.array(self.action_space)
        # deal the scaling
        raw_action_mean = action_mean.copy()
        # action_mean[1] = 1 if action_mean[1]>0 else -1 # TODO
        scale_steer = VALID_STEER[1]
        scale_speed = 1
        possible_actions = possible_actions/np.array([scale_steer, scale_speed])
        # print(possible_actions)
        prob = calculate_probability(action_mean, action_std, possible_actions)
        # prob = np.clip(prob, -10, 10)
        # prob -= prob.min()
        exp_prob = np.exp(prob) * action_mask
        prob_softmax = exp_prob / np.sum(exp_prob)
        # print(prob)
        # print(np.round(prob_softmax, 3))
        actions = np.arange(len(possible_actions))
        action_chosen = np.random.choice(actions, p=prob_softmax)
        # action_chosen = np.argmax(prob_softmax)
        # action_1 = np.clip(raw_action_mean[1], -action_mask[action_chosen], action_mask[action_chosen])
        # action_1 = np.clip(np.random.normal(raw_action_mean[1], action_std[1]),
        #                     -action_mask[action_chosen], action_mask[action_chosen])
        # possible_actions[action_chosen][1] = action_1
        # print(raw_action_mean[1], action_std[1], action_mask[action_chosen], action_1)
        return possible_actions[action_chosen]