from collections import deque

import numpy as np


class ReplayMemory(object):
    def __init__(self, memory_size: int, extra_items: list = []):
        self.items = ["state", "action", "reward", "done"] + extra_items
        self.memory = {}
        for item in self.items:
            self.memory[item] = deque([], maxlen=memory_size)
    
    def push(self, observations:tuple):
        """Save a transition"""
        for i, item in enumerate(self.items):
            self.memory[item].append(observations[i])

    def get_items(self, idx_list: np.ndarray):
        batches = {}
        for item in self.items:
            batches[item] = []
        batches["next_state"] = []
        for idx in idx_list:
            for item in self.items:
                batches[item].append(self.memory[item][idx])
            if idx == self.__len__()-1 or self.memory["done"][idx]:
                batches["next_state"].append(None)
            else:
                batches["next_state"].append(self.memory["state"][idx+1])
        for idx in batches.keys():
            if isinstance(batches[idx][0], np.ndarray):
                batches[idx] = np.array(batches[idx])
        return batches

    def sample(self, batch_size: int):
        idx_list = np.random.randint(self.__len__(), size=batch_size)
        return self.get_items(idx_list)

    def shuffle(self, idx_range: int = None):
        idx_range = self.__len__() if idx_range is None else idx_range
        idx_list = np.arange(idx_range)
        np.random.shuffle(idx_list)
        return self.get_items(idx_list)

    def clear(self):
        for item in self.items:
            self.memory[item].clear()

    def __len__(self):
        return len(self.memory["state"])