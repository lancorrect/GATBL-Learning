import sys
import os
import random

import time
from time import strftime, localtime
import argparse
import logging
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from data_utils import PollutantDataset
from trainer import Trainer
from models.GATBL_Learning import GATBL

t_start = time.time()

def get_logger(args):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir, mode=0o777)
    log_file = '{}-{}-{}.log'.format(args.model_name, args.season, strftime("%Y-%m-%d_%H:%M:%S", localtime()))
    logger.addHandler(logging.FileHandler("%s/%s" % (args.log_dir, log_file)))
    return logger

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

model_classes = {
                 'GATBL':GATBL,
                 }

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='./7cities', type=str, help='The dictionary where the data is stored.')
parser.add_argument('--season', default='spring', type=str, help='spring, summer, fall, winter')
parser.add_argument('--log_dir', default='./log', type=str, help='The dictionary to store log files.')
parser.add_argument('--adj_path', default='city_distances.csv', type=str, help='distance between each two cities')

parser.add_argument('--city_num', default=7, type=int, help='The number of cities')
parser.add_argument('--indicators_num', default=7, type=int, help='The number of cities')
parser.add_argument('--previous_hours', default=24, type=int, help='a past period used for forecast')
parser.add_argument('--next_hours', default=6, type=int, help='forecast how far in the futrue')
parser.add_argument('--threshold', default=200, type=int, help='distance threshold')
parser.add_argument('--split_rate', default=0.8, type=float)

parser.add_argument('--model_name', default='GATBL', type=str, help=', '.join(model_classes))
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--learning_rate', default=5e-3, type=float)
parser.add_argument('--weight_decay', default=5e-4, type=float)

# GAT settings
parser.add_argument('--nheads', default=32, type=int, help='graph attention heads number')
parser.add_argument('--alpha', default=0.1, type=float, help='LeakyRelu related argument')
parser.add_argument('--dropout', default=0.2, type=float)
parser.add_argument('--gat_hidden_size', default=32, type=int)

# LSTM settings
parser.add_argument('--lstm_hidden_size', default=7, type=int)
parser.add_argument('--lstm_layers', default=5, type=int)

parser.add_argument('--seed', default=1000, type=int)
parser.add_argument('--shuffle', default=True, type=bool)
parser.add_argument('--cuda', default='0', type=str, help='gpu number')
parser.add_argument('--device', default=None, type=str, help='cpu, cuda')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if args.device is None else torch.device(args.device)
print("choice cuda:{}".format(args.cuda))

logger = get_logger(args)

setup_seed(args.seed)

logger.info('\nLoading Dataset.')
trainset = PollutantDataset(args, 'train')
testset = PollutantDataset(args, 'test')
train_dataloader = DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=args.shuffle)
test_dataloader = DataLoader(dataset=testset, batch_size=args.batch_size)

logger.info('\nTrain model')
args.model_class = model_classes[args.model_name]
model = args.model_class(args).to(args.device)
optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
trainer = Trainer(args, logger, model, train_dataloader, test_dataloader,optimizer)
trainer.train()

t_end = time.time()
logger.info('Training process took '+ str(round(t_end-t_start))+ ' secs.')