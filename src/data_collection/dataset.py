import sys
sys.path.append("..")
sys.path.append(".")
import pickle
import time
import os

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# from env.car_parking_base import CarParking

class ParkingDatasetGenerator(object):
    def __init__(self,
                save_path,
                ) -> None:
        self.save_path = save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.map_level_path = None
        self.current_traj = []
        self.observation_list = []
        self.meta_data = {}

    def reset(self, map, map_level:str=None):
        self.current_traj.clear()
        self.observation_list.clear()
        self.meta_data["map_type"] = str(type(map).__name__).split('.')[-1]
        self.meta_data["map_level"] = map_level if map_level is not None else "NotDefined"
        self.map_level_path = os.path.join(self.save_path, self.meta_data["map_level"])
        if not os.path.exists(self.map_level_path):
            os.makedirs(self.map_level_path)
        self.meta_data["case_id"] = map.case_id
        self.meta_data["map"] = map

    def record(self, obs, action, reward, next_obs, done, info, rs=None):
        if len(self.observation_list) == 0:
            self.observation_list.append(obs)
        info["rs"] = rs
        data = {
            "obs_id": len(self.observation_list),
            "action": action,
            "reward": reward,
            "next_obs_id": len(self.observation_list)+1,
            "done": done,
            "info": info,
        }
        self.current_traj.append(data)
        self.observation_list.append(next_obs)

    def save_to_disk(self, other_info:dict={}):
        self.meta_data["traj_length"] = len(self.current_traj)
        self.meta_data.update(other_info)
        data = {
            "meta_data": self.meta_data,
            "data": {
                "traj": self.current_traj,
                "obs": self.observation_list,
                }
            }
        current_time = time.localtime()
        timestamp = time.strftime("%Y%m%d_%H%M%S", current_time)
        file_name = "%s_%s_%s_%s.data"%(self.meta_data["map_type"], self.meta_data["case_id"], bool(other_info["success"]), timestamp)
        file_dir = os.path.join(self.map_level_path, file_name)
        # if file already exists, rename the file
        if os.path.exists(file_dir):
            idx = 0
            while os.path.exists(file_dir):
                file_name = "%s_%s_%s_%s_%s.data"%(self.meta_data["map_type"], self.meta_data["case_id"], bool(other_info["success"]), timestamp, idx)
                file_dir = os.path.join(self.map_level_path, file_name)
                idx += 1
        pkl_file = open(file_dir, "wb")
        pickle.dump(data, pkl_file)
        pkl_file.close()

class ParkingDataset(object):
    def __init__(self,
            data_path, dataset_size = None, load_to_memory = False, load_success_only = False,
                ) -> None:
        self.data_base_path = data_path
        self.dataset_size = dataset_size
        self.load_to_memory = load_to_memory
        self.load_success_only = load_success_only
        self.data_dirs = self._init_dir()

    def _init_dir(self, ):
        t = time.time()
        print(">>> Initializing Dataset Index <<<")
        data_dirs = []
        if self.load_to_memory:
            self.traj_records = {}
        reach_dataset_size = False
        map_level_dirs = os.listdir(self.data_base_path)
        for rel_map_level_dir in map_level_dirs:
            print("> Reading %s"%rel_map_level_dir)
            map_level_dir = os.path.join(self.data_base_path, rel_map_level_dir)
            data_files = os.listdir(map_level_dir)
            for data_file in tqdm(data_files):
                data_file_dir = os.path.join(map_level_dir, data_file)
                if self.load_success_only and "False" in data_file_dir:
                    continue
                f_data = open(data_file_dir, "rb")
                traj_record = pickle.load(f_data)
                f_data.close()
                data_dir = [(data_file_dir, i) for i in range(traj_record["meta_data"]["traj_length"])]
                data_dirs.extend(data_dir)

                if self.load_to_memory:
                    self.traj_records[data_file_dir] = traj_record

                if self.dataset_size is not None and len(data_dirs) >self.dataset_size:
                    reach_dataset_size = True
                    break
            if reach_dataset_size:
                break
        if self.dataset_size is not None and len(data_dirs) >self.dataset_size:
            data_dirs = data_dirs[:self.dataset_size]
        
        print(">> Initializing Dataset Done !!")
        print(">> time consume: %s seconds"%(time.time()-t))
        return data_dirs
    
    def __len__(self,):
        return len(self.data_dirs)
    
    def __getitem__(self, idx):
        data_dir, step_idx = self.data_dirs[idx]
        if self.load_to_memory:
            traj_record = self.traj_records[data_dir]
        else:
            f_traj = open(data_dir, "rb")
            traj_record = pickle.load(f_traj)
            f_traj.close()
        step_data = traj_record["data"]["traj"][step_idx]
        all_obs_data = traj_record["data"]["obs"]
        data = {
            "obs": all_obs_data[step_idx],
            "action": step_data["action"],
            "reward": step_data["reward"],
            "next_obs": all_obs_data[step_idx+1],
            "done": step_data["done"],
            "info": step_data["info"],
        }
        return data
    
class ParkingImgDataset(object):
    def __init__(self,
            data_path, dataset_size = None, load_to_memory = False
                ) -> None:
        self.data_base_path = data_path
        self.dataset_size = dataset_size
        self.load_to_memory = load_to_memory
        self.data_dirs = self._init_dir()

    def _init_dir(self, ):
        t = time.time()
        print(">>> Initializing Dataset Index <<<")
        data_dirs = []
        if self.load_to_memory:
            self.traj_records = {}
        reach_dataset_size = False
        map_level_dirs = os.listdir(self.data_base_path)
        for rel_map_level_dir in map_level_dirs:
            print("> Reading %s"%rel_map_level_dir)
            map_level_dir = os.path.join(self.data_base_path, rel_map_level_dir)
            data_files = os.listdir(map_level_dir)
            for data_file in tqdm(data_files):
                data_file_dir = os.path.join(map_level_dir, data_file)
                f_data = open(data_file_dir, "rb")
                traj_record = pickle.load(f_data)
                f_data.close()
                data_dir = [(data_file_dir, i) for i in range(traj_record["meta_data"]["traj_length"]+1)]
                data_dirs.extend(data_dir)

                if self.load_to_memory:
                    self.traj_records[data_file_dir] = traj_record

                if self.dataset_size is not None and len(data_dirs) >self.dataset_size:
                    reach_dataset_size = True
                    break
            if reach_dataset_size:
                break
        if self.dataset_size is not None and len(data_dirs) >self.dataset_size:
            data_dirs = data_dirs[:self.dataset_size]
        
        print(">> Initializing Dataset Done !!")
        print(">> time consume: %s seconds"%(time.time()-t))
        return data_dirs
    
    def __len__(self,):
        return len(self.data_dirs)
    
    def __getitem__(self, idx):
        data_dir, step_idx = self.data_dirs[idx]
        if self.load_to_memory:
            traj_record = self.traj_records[data_dir]
            img = traj_record["data"]["obs"][step_idx]["img"]
        else:
            f_traj = open(data_dir, "rb")
            traj_record = pickle.load(f_traj)
            f_traj.close()
            img = traj_record["data"]["obs"][step_idx]["img"]
        return img
    
    def get_img_shape(self, ):
        data_dir, step_idx = self.data_dirs[0]
        f_traj = open(data_dir, "rb")
        traj_record = pickle.load(f_traj)
        f_traj.close()
        img = traj_record["data"]["obs"][step_idx]["img"]
        return img.shape

if __name__ == "__main__":
    # parking_dataset = ParkingDataset("./agent_data/use_img")
    # print(len(parking_dataset))
    # print(parking_dataset[10])
    # t = time.time()
    # for _ in range(20):
    #     idx = np.random.randint(0,len(parking_dataset))
    #     data = parking_dataset[idx]
    # print(time.time()-t)

    parking_dataset = ParkingImgDataset("./agent_data/use_img")
    print(len(parking_dataset))
    print(parking_dataset[10])
    t = time.time()
    for _ in range(20):
        idx = np.random.randint(0,len(parking_dataset))
        data = parking_dataset[idx]
    print(time.time()-t)