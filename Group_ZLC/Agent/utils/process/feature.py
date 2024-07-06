import numpy
import json
# JSON File Path
file_path = 'utils/process/config.json'
# Load JSON File
with open(file_path, 'r') as f:
    scale = json.load(f)
for key in scale.keys():
    scale[key] = float(scale[key])
# min_position, max_position
# min_v, max_v
# min_a, max_a
# min_j, max_j
def feature_process(data):
    #[batch_size, token_num, token_size]
    #x, y, v, a, j, 1.x, 1.y, 2.x, 2.y, 3.x, 3.y, 4.x, 4.y, 5.x, 5.y, 6.x, 6.y, obs.x, obs.y, isTOR 
    ### position
    ###
    idx_odds = [0, 5, 7, 9, 11, 13, 15, 17]
    idx_evens = [1, 6, 8, 10, 12, 14, 16, 18]
    obs_x = data[:, :, 17]
    obs_y = data[:, :, 18]
    for idx in idx_odds:
        data[:, :, idx] -= obs_x
    for idx in idx_evens:
        data[:, :, idx] -= obs_y
    data[:, :, : 2]  = data[:, :, : 2] / (scale['max_position'] - scale['min_position'])
    data[:, :, 5 : 19]  = data[:, :, 5 : 19] / (scale['max_position'] - scale['min_position'])
    ### v
    data[:, :, 2] = (data[:, :, 2] - scale['min_v']) / (scale['max_v'] - scale['min_v'])
    ### a
    data[:, :, 3] = (data[:, :, 3] - scale['min_a']) / (scale['max_a'] - scale['min_a'])
    ### j
    data[:, :, 4] = (data[:, :, 4] - scale['min_j']) / (scale['max_j'] - scale['min_j'])
    return data

def reward_process(reward):
    reward = reward / scale['reward_scale']
    return reward


def feature_reverse(data):
    #[batch_size, token_num, token_size]
    #x, y, v, a, j, 1.x, 1.y, 2.x, 2.y, 3.x, 3.y, 4.x, 4.y, 5.x, 5.y, 6.x, 6.y, obs.x, obs.y, isTOR 
    ### position    
    data[:, :, : 2]  = data[:, :, : 2] * (scale['max_position'] - scale['min_position'])
    data[:, :, 5 : 19]  = data[:, :, 5 : 19] * (scale['max_position'] - scale['min_position'])
    ### v
    data[:, :, 2] = data[:, :, 2] * (scale['max_v'] - scale['min_v']) + scale['min_v']
    ### a
    data[:, :, 3] = data[:, :, 3] * (scale['max_a'] - scale['min_a']) + scale['min_a']
    ### j
    data[:, :, 4] = data[:, :, 4] * (scale['max_j'] - scale['min_j']) + scale['min_j']
    return data

def reward_reverse(reward):
    reward = reward * scale['reward_scale']
    return reward