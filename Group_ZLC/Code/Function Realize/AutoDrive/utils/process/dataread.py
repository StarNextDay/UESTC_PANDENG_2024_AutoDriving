import pandas as pd
import numpy as np
from utils.process.feature import feature_process, reward_process
import os
import json
class Data:
    def __init__(self):
        file_path = 'utils/process/data.json'
        # Load JSON File
        with open(file_path, 'r') as f:
            data = json.load(f)
        rewards_path = data['reward']
        data_path = data['preprocess_data']
        self.file_names = []
        self.file_rewards = []
        with open(rewards_path, 'r') as file:
            for line in file.readlines():
                line = line.split(' ')
                self.file_names.append(line[0] + ' ' + line[1] + '.csv')
                self.file_rewards.append(float(line[2]))
        self.rewards = []
        self.datas = []
        for data_dirs in os.listdir(data_path):
            for data_dir in os.listdir(data_path + '/' + data_dirs):
                idx = self.file_names.index(data_dir)
                tmp_data = np.array(pd.read_csv(data_path + '/' + data_dirs + '/' + data_dir))
                self.rewards.append(self.file_rewards[idx])
                self.datas.append(tmp_data)
        self.rewards = reward_process(np.array(self.rewards))
        self.datas = feature_process(np.array(self.datas))
        self.size = np.shape(self.rewards)[0]
    def get_datas(self, batch_size = 4):
        end = np.shape(self.rewards)[0]
        indexs = np.random.randint(0, end, [batch_size])
        return self.rewards[indexs], self.datas[indexs]