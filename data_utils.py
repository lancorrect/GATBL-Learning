import os
import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset
import pandas as pd
from tqdm import tqdm
from math import floor

def load_dataset(args):
    data = []
    data_dir = os.path.join(args.data_dir, args.season)
    files = os.listdir(data_dir)
    files.sort()
    for f in files:
        path = os.path.join(data_dir, f)
        data_city = pd.read_csv(path)
        data.append(np.array(data_city, dtype=np.float32))
    data = np.array(data, dtype=np.float32)
    return data

def get_adj(args):
    adj_file = np.array(pd.read_csv(args.adj_path))[:, 1:]
    threshold = args.threshold
    adj = np.where(adj_file < threshold, 1.0, 0.0)
    return adj.astype(np.float32)

class PollutantDataset(Dataset):
    def __init__(self, args, split) -> list:
        data=[]

        data_origin = load_dataset(args)
        data_origin = data_origin.transpose((1, 0, 2))
        boundary = floor(len(data_origin)* args.split_rate)

        data_origin = data_origin[:boundary, :, :] if split == 'train' else data_origin[boundary:, :, :]

        how_pre_hours = args.previous_hours
        how_far_hours = args.next_hours
        hour_end = len(data_origin)-how_far_hours
        hour_range = range(how_pre_hours, hour_end)

        adj = get_adj(args)

        desc_str = 'Training examples' if split=='train' else 'Testing examples'
        for index in tqdm(hour_range, total=len(hour_range), desc=desc_str):
            data_previous = data_origin[ index-how_pre_hours:index, :, :]
            data_next = data_origin[index, :, 1]
            data.append({'previous': data_previous, 'next': data_next, 'adj':adj})
        
        self._data = data

    def __getitem__(self, index):
        return self._data[index]
    
    def __len__(self):
        return len(self._data)
    

def restored_pm25(args, nor):

    mean_dict = np.load('./7cities/mean.npy', allow_pickle='TRUE').item()[args.season]
    std_dict = np.load('./7cities/std.npy', allow_pickle='TRUE').item()[args.season]
    mean_list = []
    std_list = []

    for values in mean_dict.values():
        mean_list += [values[1]]
    for values in std_dict.values():
        std_list += [values[1]]
    
    mean_list = torch.tensor(mean_list).unsqueeze(0).transpose(1, 0).to(args.device)
    std_list = torch.tensor(std_list).unsqueeze(0).transpose(1, 0).to(args.device)
    result = torch.mul(nor, std_list) + mean_list
    return result
