import os
import sys
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from time import strftime, localtime
import copy

from data_utils import restored_pm25

class Trainer:
    def __init__(self, args, logger, model, train_dataloader, test_dataloader, optimizer) -> None:
        self.args = args
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.logger = logger
        self.optimizer = optimizer

        self.logger.info('training arguments:')

        for arg in vars(self.args):
            self.logger.info('>>> {0}: {1}'.format(arg, getattr(self.args, arg)))

    def train(self):

        best_loss = np.inf
        best_mse, best_rmse, best_mae = None, None, None
        self.criterion_mse = nn.MSELoss()
        self.criterion_smoothl1 = nn.SmoothL1Loss()

        for epoch in range(self.args.epochs):

            self.logger.info('>' * 60)
            self.logger.info('epoch: {}'.format(epoch+1))
            self.model.train()

            loss = 0
            outputs_all = torch.zeros([self.args.city_num, self.args.next_hours])
            target_all = torch.zeros([self.args.city_num, self.args.next_hours])

            for index, sample_batch in enumerate(tqdm(self.train_dataloader, desc='Train')):

                inputs = sample_batch['previous'].to(self.args.device)
                adj = sample_batch['adj'].to(self.args.device)
                outputs = self.model(inputs, adj)
                target = sample_batch['next'].to(self.args.device).transpose(1, 0)
                outputs_all[:, (index % self.args.next_hours)] = outputs[:, 0]
                target_all[:, (index % self.args.next_hours)] = target[:, 0]
                
                if (index + 1) % self.args.next_hours == 0:
                    
                    loss_mse = self.criterion_mse(outputs_all, target_all)
                    loss_l1 = self.criterion_smoothl1(outputs_all, target_all)
                    loss = loss_mse + loss_l1

                    loss.backward()
                    self.optimizer.step()

                    self.optimizer.zero_grad()
                    loss = 0
                    outputs_all = torch.zeros([self.args.city_num, self.args.next_hours])
                    target_all = torch.zeros([self.args.city_num, self.args.next_hours])

            test_loss, mse, rmse, mae= self.evaluate()
            if test_loss < best_loss:
                
                best_loss = test_loss
                best_mse = mse
                best_rmse = rmse
                best_mae = mae

                if not os.path.exists('./best_model'):
                    os.mkdir('./best_model')
                model_path = './best_model/{}_{}_mse_{:.2f}_rmse_{:.2f}_mae_{:.2f}'.format(self.args.season,
                                                                        self.args.model_name, mse, rmse, mae)
                self.best_model = copy.deepcopy(self.model)
                self.logger.info('>> saved:{}'.format(model_path))
        
        self.logger.info('>' * 60)
        self.logger.info('save best model')
        torch.save(self.best_model, model_path)
        self.logger.info(
            'mse: {:.2f}, rmse: {:.2f}, mae: {:.2f}'.format(best_mse, best_rmse, best_mae))

    def evaluate(self):

        loss_all, mse, rmse, mae = [], [], [], []
        criterion_mae = nn.L1Loss()
        self.model.eval()

        with torch.no_grad():

            outputs_all = torch.zeros([self.args.city_num, self.args.next_hours])
            target_all = torch.zeros([self.args.city_num, self.args.next_hours])

            for index, sample_batch in enumerate(tqdm(self.test_dataloader, desc='Test')):
                self.model.train()

                inputs = sample_batch['previous'].to(self.args.device)
                adj = sample_batch['adj'].to(self.args.device)
                outputs = self.model(inputs, adj)
                target = sample_batch['next'].to(self.args.device).transpose(1, 0)

                outputs = restored_pm25(self.args, outputs)  # [city_num, 1]
                target = restored_pm25(self.args, target) # [city_num, 1]
                outputs_all[:, (index % self.args.next_hours)] = outputs[:, 0]
                target_all[:, (index % self.args.next_hours)] = target[:, 0]

                if (index+1) % self.args.next_hours == 0:
                    loss_mse = self.criterion_mse(outputs_all, target_all)
                    loss_rmse = torch.sqrt(loss_mse)
                    loss_mae = criterion_mae(outputs_all, target_all)
                    loss_l1 = self.criterion_smoothl1(outputs_all, target_all)
                    loss = loss_mse + loss_l1

                    loss_all.append(loss.item())
                    mse.append(loss_mse.item())
                    rmse.append(loss_rmse.item())
                    mae.append(loss_mae.item())

                    loss = 0
                    outputs_all = torch.zeros([self.args.city_num, self.args.next_hours])
                    target_all = torch.zeros([self.args.city_num, self.args.next_hours])
        
        loss_all = np.array(loss_all).mean()
        mse = np.array(mse).mean()
        rmse = np.array(rmse).mean()
        mae = np.array(mae).mean()

        return loss_all, mse, rmse, mae